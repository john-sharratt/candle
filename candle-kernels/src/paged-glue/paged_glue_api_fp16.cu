// =============================================================================
// paged_glue_api_fp16.cu — reprojection glue forward dispatch (FP16).
//
// The glue kernel is a derivative of the INT8 paged-decode batched-M kernel,
// widened from one query token to G glue tokens per slot. It streams each
// quantized prefix chunk once and reuses it across all G x hpg query rows via
// the tensor-core M-dimension (dequant-once). HEAD_DIM=128 is the production
// path (SUB_HEAD_DIM==32, the m16n8k32 palette width); other head dims route to
// the plain prefill on the host side.
// =============================================================================

#include "paged_glue_kernel.cuh"

#include <cuda_fp16.h>

extern "C" void run_paged_glue_fp16(
    const void* q_ptr,
    const uint8_t* headers_ptr,
    void* o_ptr,
    int32_t batch,
    int32_t max_glue,
    int32_t n_q_head,
    int32_t n_kv_head,
    int32_t head_dim,
    float softmax_scale,
    const void* k_new,
    const void* v_new,
    const float* rope_cs,
    int32_t rope_interleaved,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    const uint32_t* col_actual_pos,
    const uint32_t* cu_kvlens,
    const uint32_t* glue_write_slice,
    const uint32_t* glue_write_in_blk,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    #define LAUNCH_GLUE(HD) \
        paged_glue::launch_paged_glue_attn<__half, __half, __half, HD>( \
            (const __half*)q_ptr, headers_ptr, (__half*)o_ptr, \
            batch, max_glue, n_q_head, n_kv_head, softmax_scale, \
            (const __half*)k_new, (const __half*)v_new, rope_cs, rope_interleaved, \
            cu_seqlens_q, q_lens, kv_lens, col_actual_pos, cu_kvlens, \
            glue_write_slice, glue_write_in_blk, stream)
    switch (head_dim) {
        case 128: LAUNCH_GLUE(128); break;
        default: break;
    }
    #undef LAUNCH_GLUE
}
