#pragma once
// =============================================================================
// v2_compat_dispatch.cuh — v2-API-compatible entry point.
//
// Routes through `launch_int8_decode_attn` (our owned kernel that we will
// modify in place toward INT8 MMA). Falling back to v2 if needed is the
// caller's responsibility (Rust env-var hook).
// =============================================================================

#include "int8_decode_kernel.cuh"

namespace fused_attn {

template<typename Q_T, typename T, typename O, int HEAD_DIM>
cudaError_t launch_v2_compat(
    const Q_T*     q,
    const uint8_t* headers_ptr,
    O*             out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const T*       k_new,
    const T*       v_new,
    const float*   rope_cs,
    int            rope_interleaved,
    cudaStream_t   stream
) {
    launch_int8_decode_attn<Q_T, T, O, HEAD_DIM>(
        q, headers_ptr, out,
        num_active_slots, n_q_head, n_kv_head,
        softmax_scale, k_new, v_new, rope_cs, rope_interleaved, stream);
    return cudaGetLastError();
}

} // namespace fused_attn
