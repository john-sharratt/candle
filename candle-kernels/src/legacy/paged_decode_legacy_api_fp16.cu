#include "paged_decode_kernel.cuh"

// API — FP16 dtype
extern "C" void run_paged_decode_legacy_fp16(
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
    #define LAUNCH_FP16(HD) \
        launch_paged_decode_attn<__half, __half, __half, HD>( \
            (const __half*)q_ptr, headers_ptr, (__half*)o_ptr, \
            num_active_slots, n_q_head, n_kv_head, softmax_scale, \
            (const __half*)k_new, (const __half*)v_new, rope_cs, rope_interleaved, stream)
    switch (head_dim) {
        case 64:  LAUNCH_FP16(64);  break;
        case 96:  LAUNCH_FP16(96);  break;
        case 128: LAUNCH_FP16(128); break;
        case 256: LAUNCH_FP16(256); break;
        default: break;
    }
    #undef LAUNCH_FP16
}
