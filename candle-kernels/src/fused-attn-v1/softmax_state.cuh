#pragma once
// =============================================================================
// softmax_state.cuh — online (Flash-Attn-2 style) softmax state.
//
// Wraps the m_i / l_i / alpha / beta pattern from v2's process_tile inline
// loop into a struct so each consumer warp's softmax state is register-resident
// with clear lifetime.
// =============================================================================

#include "../fast_exp.cuh"
#include "../simple/warp_reduce.cuh"
#include <cuda_runtime.h>

namespace fused_attn {

// warp-collective max-reduction (warp_reduce.cuh only ships warp_reduce_sum).
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

struct OnlineSoftmaxState {
    float m_i;   // running max
    float l_i;   // running sum of exps

    __device__ __forceinline__ void init() {
        m_i = -1e38f;
        l_i = 0.f;
    }

    template<int N_LOGITS_PER_THREAD>
    __device__ __forceinline__ float update(float (&logits)[N_LOGITS_PER_THREAD]) {
        // Step 1: tile-local max across this thread's logits, reduced across warp.
        float tile_max = -1e38f;
        #pragma unroll
        for (int e = 0; e < N_LOGITS_PER_THREAD; ++e) {
            if (logits[e] > tile_max) tile_max = logits[e];
        }
        tile_max = warp_reduce_max(tile_max);

        // Step 2: combine with running max + handle first-tile sentinel.
        float new_m = fmaxf(m_i, tile_max);
        float alpha;
        if (m_i < -1e30f) {
            alpha = 0.f;
        } else {
            alpha = fast_exp::exp<float, fast_exp::Softmax>(m_i - new_m);
        }

        // Step 3: replace logits with exp(logit - new_m), accumulate sum.
        float tile_sum = 0.f;
        #pragma unroll
        for (int e = 0; e < N_LOGITS_PER_THREAD; ++e) {
            float p = fast_exp::exp<float, fast_exp::Softmax>(logits[e] - new_m);
            logits[e] = p;
            tile_sum += p;
        }
        tile_sum = warp_reduce_sum(tile_sum);

        // Step 4: update running state.
        l_i = l_i * alpha + tile_sum;
        m_i = new_m;

        return alpha;
    }

    __device__ __forceinline__ float normalizer() const {
        return __fdividef(1.f, fmaxf(l_i, 1e-10f));
    }
};

} // namespace fused_attn
