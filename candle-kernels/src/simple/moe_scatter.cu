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
