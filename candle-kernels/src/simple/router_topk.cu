// =============================================================================
// FUSED MoE ROUTER EPILOGUE (elementwise score functions)
// =============================================================================
// One launch replacing the router's eager-op chain after the gate GEMM:
// score(logits) → +bias → descending top-k → gather unbiased scores →
// normalize → ×route_scale. The eager path was ~15 elementwise/sort/gather
// launches per MoE layer per wave — pure per-launch submission tax on WDDM,
// where a launch costs ~67 µs of host time regardless of its GPU work.
//
// Scope: ELEMENTWISE score functions only (sigmoid, sqrt-softplus). Softmax
// scores need a cross-expert reduction whose summation order the eager path
// fixes; callers keep the eager chain for it.
//
// Bit-exactness contract with the eager path (the gates verify it end-to-end):
//   * sqrt-softplus reproduces the numerically-stable eager op sequence
//     exactly: sqrtf(relu(x) + logf(expf(-fabsf(x)) + 1))
//   * sigmoid reproduces 1 / (1 + expf(-x))
//   * selection is descending by (score + bias), ties broken by LOWER expert
//     id (the eager arg_sort's observable order on real, tie-free scores)
//   * the normalizer sums the k selected scores IN SELECTION ORDER (the eager
//     `sum` over the arg_sort-ordered row), then each weight is
//     (w / denom) * route_scale — div before scale, like the eager chain.
//
// One WARP per token row: lanes stride the expert axis (coalesced), the top-k
// loop runs k warp-argmax reductions (value desc, index asc), and the owning
// lane broadcasts the winner's unbiased score. k ≤ 16, n_experts ≤ 32·32.

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "../fast_exp.cuh"

#define RT_MAX_PER_LANE 32     // n_experts ≤ 32 lanes × 32 = 1024
#define RT_MAX_TOPK 16

// Score function selectors (mirrored by the Rust wrapper).
#define RT_SCORE_SIGMOID 1
#define RT_SCORE_SQRT_SOFTPLUS 2

__device__ __forceinline__ float rt_score(float x, int func) {
    if (func == RT_SCORE_SIGMOID) {
        // The eager path's `usigmoid_f32` kernel — the same fast_exp
        // implementation, for bit-identical scores.
        return fast_exp::sigmoid<float>(x);
    }
    // Numerically-stable softplus, the eager path's exact op sequence AND
    // intrinsics: relu(x) + ln(1 + e^-|x|), then sqrt — where the eager `exp`
    // is `fast_exp::exp` (`uexp_f32`) and `log`/`sqrt` are the standard
    // `logf`/`sqrtf` (`ulog_f32`/`usqrt_f32`).
    float stable = logf(fast_exp::exp<float>(-fabsf(x)) + 1.0f);
    float r = x > 0.0f ? x : 0.0f;
    return sqrtf(r + stable);
}

__global__ void router_topk_kernel(
    const float* __restrict__ logits, // [nt, ne]
    const float* __restrict__ bias,   // [ne] or nullptr
    int ne,
    int k,
    int func,
    float route_scale,
    float* __restrict__ out_w,   // [nt, k]
    uint32_t* __restrict__ out_i // [nt, k]
) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float* lrow = logits + (size_t)row * ne;

    // Per-lane strided load + elementwise score; `sel` is what the top-k
    // orders by, `sc` what the weights gather.
    float sc[RT_MAX_PER_LANE];
    float sel[RT_MAX_PER_LANE];
    const int per_lane = (ne + 31) / 32;
#pragma unroll
    for (int j = 0; j < RT_MAX_PER_LANE; ++j) {
        if (j >= per_lane) break;
        const int e = lane + 32 * j;
        if (e < ne) {
            const float s = rt_score(lrow[e], func);
            sc[j] = s;
            sel[j] = bias ? s + bias[e] : s;
        } else {
            sc[j] = 0.0f;
            sel[j] = -INFINITY;
        }
    }

    float picked_w[RT_MAX_TOPK];
    uint32_t picked_i[RT_MAX_TOPK];
    for (int t = 0; t < k; ++t) {
        // Lane-local best (value desc, index asc).
        float best_v = -INFINITY;
        int best_j = -1;
#pragma unroll
        for (int j = 0; j < RT_MAX_PER_LANE; ++j) {
            if (j >= per_lane) break;
            if (sel[j] > best_v) {
                best_v = sel[j];
                best_j = j;
            }
        }
        int best_e = best_j < 0 ? INT_MAX : lane + 32 * best_j;
        // Warp reduce: max value, then min expert id on ties.
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, best_v, off);
            const int oe = __shfl_down_sync(0xffffffffu, best_e, off);
            if (ov > best_v || (ov == best_v && oe < best_e)) {
                best_v = ov;
                best_e = oe;
            }
        }
        best_e = __shfl_sync(0xffffffffu, best_e, 0);
        // The owner lane broadcasts the winner's UNBIASED score and retires it.
        const int win_lane = best_e % 32;
        const int win_j = best_e / 32;
        float w = 0.0f;
        if (lane == win_lane) {
            w = sc[win_j];
            sel[win_j] = -INFINITY;
        }
        w = __shfl_sync(0xffffffffu, w, win_lane);
        picked_w[t] = w;
        picked_i[t] = (uint32_t)best_e;
    }

    if (lane == 0) {
        // The normalizer reproduces `fast_sum`'s pairwise tree over the
        // arg_sort-ordered row EXACTLY (the eager `sum_keepdim`): pad to the
        // next power of two with zeros and halve — for k = 8 that is
        // ((w0+w4)+(w2+w6)) + ((w1+w5)+(w3+w7)), which differs from a
        // sequential sum by 1 ulp and would break the bit-exactness contract.
        float shr[2 * RT_MAX_TOPK];
        int p = 1;
        while (p < k) p <<= 1;
        for (int t = 0; t < p; ++t) {
            shr[t] = t < k ? picked_w[t] : 0.0f;
        }
        for (int s = p / 2; s > 0; s >>= 1) {
            for (int i = 0; i < s; ++i) {
                shr[i] += shr[i + s];
            }
        }
        const float denom = shr[0];
        for (int t = 0; t < k; ++t) {
            out_w[(size_t)row * k + t] = (picked_w[t] / denom) * route_scale;
            out_i[(size_t)row * k + t] = picked_i[t];
        }
    }
}

extern "C" int run_router_topk(
    const void* logits,
    const void* bias, // nullptr when the gate has no bias
    int32_t n_tokens,
    int32_t n_experts,
    int32_t k,
    int32_t score_func,
    float route_scale,
    void* out_w,
    void* out_i,
    void* stream)
{
    if (n_tokens <= 0 || n_experts <= 0 || n_experts > 32 * RT_MAX_PER_LANE ||
        k <= 0 || k > RT_MAX_TOPK || k > n_experts ||
        (score_func != RT_SCORE_SIGMOID && score_func != RT_SCORE_SQRT_SOFTPLUS)) {
        return -1;
    }
    router_topk_kernel<<<n_tokens, 32, 0, (cudaStream_t)stream>>>(
        (const float*)logits, (const float*)bias, n_experts, k, score_func,
        route_scale, (float*)out_w, (uint32_t*)out_i);
    return (int)cudaGetLastError();
}
