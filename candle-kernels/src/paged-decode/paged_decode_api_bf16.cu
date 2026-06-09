// =============================================================================
// paged_decode_api_bf16.cu — default decode dispatch (BF16).
//
// The INT8 decode kernel (split-KV / warp-stripe / batched-M) is the production
// path for head_dim 64/96/128. head_dim 256 falls back to the legacy V2 kernel.
// =============================================================================

#include "int8_decode_kernel.cuh"
#include "../legacy/paged_decode_kernel.cuh"  // launch_paged_decode_attn (hd256 fallback)

#include <cuda_bf16.h>

extern "C" void run_paged_decode_bf16(
    const void* q_ptr,
    const uint8_t* headers_ptr,
    void* o_ptr,
    int32_t num_active_slots,
    int32_t n_q_head,
    int32_t n_kv_head,
    int32_t head_dim,
    float softmax_scale,
    const void* k_new,
    const void* v_new,
    const float* rope_cs,
    int32_t rope_interleaved,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    #define LAUNCH_INT8(HD) \
        fused_attn::launch_int8_decode_attn<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, HD>( \
            (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr, \
            num_active_slots, n_q_head, n_kv_head, softmax_scale, \
            (const __nv_bfloat16*)k_new, (const __nv_bfloat16*)v_new, rope_cs, rope_interleaved, stream)
    switch (head_dim) {
        case 64:  LAUNCH_INT8(64);  break;
        case 96:  LAUNCH_INT8(96);  break;
        case 128: LAUNCH_INT8(128); break;
        case 256:
            launch_paged_decode_attn<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, 256>(
                (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr,
                num_active_slots, n_q_head, n_kv_head, softmax_scale,
                (const __nv_bfloat16*)k_new, (const __nv_bfloat16*)v_new, rope_cs, rope_interleaved, stream);
            break;
        default: break;
    }
    #undef LAUNCH_INT8
}
