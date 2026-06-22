// =============================================================================
// paged_decode_api_fp16.cu — default decode dispatch (FP16).
//
// The INT8 decode kernel (split-KV / warp-stripe / batched-M) is the production
// path for head_dim 64/96/128/256. head_dim 256 runs its wide (hpg>8) path
// single-stage so the tiles fit the 48 KiB static shared-memory cap; the stripe
// and batched-M paths are unchanged.
// =============================================================================

#include "int8_decode_kernel.cuh"

#include <cuda_fp16.h>

extern "C" void run_paged_decode_fp16(
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
        fused_attn::launch_int8_decode_attn<__half, __half, __half, HD>( \
            (const __half*)q_ptr, headers_ptr, (__half*)o_ptr, \
            num_active_slots, n_q_head, n_kv_head, softmax_scale, \
            (const __half*)k_new, (const __half*)v_new, rope_cs, rope_interleaved, stream)
    switch (head_dim) {
        case 64:  LAUNCH_INT8(64);  break;
        case 96:  LAUNCH_INT8(96);  break;
        case 128: LAUNCH_INT8(128); break;
        case 256: LAUNCH_INT8(256); break;
        default: break;
    }
    #undef LAUNCH_INT8
}

// B2: decode with fused q8a128 context output (feeds o_proj directly, no standalone
// quantize). Only head_dim 128, where the combine block is exactly one q8a128 tile.
extern "C" void run_paged_decode_fp16_q8(
    const void* q_ptr,
    const uint8_t* headers_ptr,
    void* q8_out,
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
    if (head_dim != 128) return; // q8a128 output supported only at head_dim 128
    fused_attn::launch_int8_decode_attn<__half, __half, __half, 128>(
        (const __half*)q_ptr, headers_ptr, (__half*)nullptr,
        num_active_slots, n_q_head, n_kv_head, softmax_scale,
        (const __half*)k_new, (const __half*)v_new, rope_cs, rope_interleaved,
        stream, (uint8_t*)q8_out);
}
