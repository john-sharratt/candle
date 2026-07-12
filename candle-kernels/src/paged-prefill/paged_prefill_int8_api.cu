#include <stdint.h>
#include <cuda_runtime.h>
#include "../arena_table.cuh" // ArenaFormat dtype codes

// ============================================================================
// INT8 Prefix-Attention Prefill API — Unified Dispatcher
// q_dtype selects the per-dtype entry (1 = F16, 2 = BF16 — the ArenaFormat
// dtype codes shared with the FP16 prefill dispatcher). Unsupported dtypes
// are a hard error, not a silent no-op.
// ============================================================================

extern "C" void run_paged_prefill_int8_fp16(
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, const uint32_t*, const float*, int32_t, cudaStream_t);

extern "C" void run_paged_prefill_int8_bf16(
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, const uint32_t*, const float*, int32_t, cudaStream_t);

extern "C" void run_paged_prefill_int8(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    void* o_ptr,
    int32_t total_q,
    int32_t batch_size,
    int32_t n_head,
    int32_t n_kv_head,
    int32_t head_dim,
    int32_t max_q_len,
    float softmax_scale,
    int32_t q_dtype,
    const uint32_t* rope_offsets,
    const float* rope_cs,
    int32_t rope_interleaved,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    switch (q_dtype) {
        case ArenaFormat::F16:
            run_paged_prefill_int8_fp16(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, head_dim,
                max_q_len, softmax_scale, rope_offsets, rope_cs,
                rope_interleaved, stream);
            break;
        case ArenaFormat::BF16:
            run_paged_prefill_int8_bf16(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, head_dim,
                max_q_len, softmax_scale, rope_offsets, rope_cs,
                rope_interleaved, stream);
            break;
        default:
            fprintf(stderr, "run_paged_prefill_int8: unsupported q_dtype %d\n", q_dtype);
            break;
    }
}
