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
// Weighted scatter-add:
//   ys[token_ids[i], j] += weights_flat[weight_ids[i]] * src[i, j]
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

// B3: byte-row gather for pre-quantized q8a1024 activations. The q8a1024 layout is
// token-contiguous when hidden % 1024 == 0 (each token occupies hidden/1024 super-blocks of
// 1152 B), so gathering quantized tokens is a plain byte copy: `hidden_dim` here is the
// per-token byte count (hidden/1024 * 1152), `xs`/`out` are the q8a1024 byte buffers. Lets the
// experts consume the already-quantized router input directly — no gather-then-quantize.
extern "C" __global__ void moe_gather_u8(
    uint8_t* out, const uint8_t* xs,
    const uint32_t* token_ids,
    size_t total_rows, size_t row_bytes
) {
    moe_gather_impl(out, xs, token_ids, total_rows, row_bytes);
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
// ACCUMULATES into ys (+=). Initialize ys to zero before the first call.

extern "C" __global__ void deterministic_scatter_bf16(
    __nv_bfloat16* __restrict__ ys,
    const __nv_bfloat16* __restrict__ down_out,
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
    __nv_bfloat16* dst = ys + (size_t)t * hidden;
    for (int col = (int)(blockIdx.y * blockDim.x + threadIdx.x);
         col < hidden;
         col += (int)(blockDim.x * gridDim.y)) {
        float sum = __bfloat162float(dst[col]);
        for (int idx = start; idx < end; idx++) {
            float w = weights_flat[reordered_weight_ids[idx]];
            float v = __bfloat162float(down_out[(size_t)perm[idx] * hidden + col]);
            sum += w * v;
        }
        dst[col] = __float2bfloat16(sum);
    }
}

extern "C" __global__ void deterministic_scatter_f16(
    __half* __restrict__ ys,
    const __half* __restrict__ down_out,
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
    __half* dst = ys + (size_t)t * hidden;
    for (int col = (int)(blockIdx.y * blockDim.x + threadIdx.x);
         col < hidden;
         col += (int)(blockDim.x * gridDim.y)) {
        float sum = __half2float(dst[col]);
        for (int idx = start; idx < end; idx++) {
            float w = weights_flat[reordered_weight_ids[idx]];
            float v = __half2float(down_out[(size_t)perm[idx] * hidden + col]);
            sum += w * v;
        }
        dst[col] = __float2half(sum);
    }
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
    const int t = (int)blockIdx.x;
    if (t >= num_tokens) return;
    const int start = token_starts[t];
    const int end   = token_starts[t + 1];
    float* dst = ys + (size_t)t * hidden;
    for (int col = (int)(blockIdx.y * blockDim.x + threadIdx.x);
         col < hidden;
         col += (int)(blockDim.x * gridDim.y)) {
        float sum = dst[col];
        for (int idx = start; idx < end; idx++) {
            float w = weights_flat[reordered_weight_ids[idx]];
            sum += w * down_out[(size_t)perm[idx] * hidden + col];
        }
        dst[col] = sum;
    }
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
#define MOE_ROUTE_MAX_SLOTS 8   // experts per lane ⇒ supports up to 32·8 = 256 experts

template<typename T>
__device__ __forceinline__ float moe_route_to_f32(T x);
template<> __device__ __forceinline__ float moe_route_to_f32<float>(float x) { return x; }
template<> __device__ __forceinline__ float moe_route_to_f32<__half>(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float moe_route_to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T>
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
    float v[MOE_ROUTE_MAX_SLOTS];
    #pragma unroll
    for (int j = 0; j < MOE_ROUTE_MAX_SLOTS; ++j) {
        int e = lane + 32 * j;
        v[j] = (e < n_experts) ? moe_route_to_f32<T>(row[e]) : -INFINITY;
    }

    // Global max (warp reduction over the per-lane local max).
    float gmax = -INFINITY;
    #pragma unroll
    for (int j = 0; j < MOE_ROUTE_MAX_SLOTS; ++j) gmax = fmaxf(gmax, v[j]);
    for (int off = 16; off > 0; off >>= 1) gmax = fmaxf(gmax, __shfl_xor_sync(FULL, gmax, off));

    // Full Σ exp(l − gmax) — only needed for the un-renormalized softmax weights.
    float z_all = 0.f;
    if (!norm_topk) {
        float ls = 0.f;
        #pragma unroll
        for (int j = 0; j < MOE_ROUTE_MAX_SLOTS; ++j) {
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
        for (int j = 0; j < MOE_ROUTE_MAX_SLOTS; ++j) {
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
        if (lane == (bi & 31)) v[bi >> 5] = -INFINITY;  // owner masks the winner
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
    moe_route_impl<float>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_f16(
    const __half* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__half>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_bf16(
    const __nv_bfloat16* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__nv_bfloat16>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}
