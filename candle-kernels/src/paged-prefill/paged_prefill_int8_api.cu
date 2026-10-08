#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "../arena_table.cuh" // ArenaFormat dtype codes
#include "../rope/rope_table.cuh" // RopeRungs

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
    float, const RopeRungs, int32_t, cudaStream_t,
    const uint32_t*, const uint32_t*, const uint2*, const uint2*, int32_t, int32_t,
    uint8_t*, int64_t, int64_t, int32_t, int32_t);

extern "C" void run_paged_prefill_int8_bf16(
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, const RopeRungs, int32_t, cudaStream_t,
    const uint32_t*, const uint32_t*, const uint2*, const uint2*, int32_t, int32_t,
    uint8_t*, int64_t, int64_t, int32_t, int32_t);

extern "C" void run_paged_prefill_kv_stage_fp16(
    const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*,
    int32_t, int32_t, int32_t, const RopeRungs, int32_t, cudaStream_t,
    uint8_t*, int64_t, int64_t, int32_t, int32_t);

extern "C" void run_paged_prefill_kv_stage_bf16(
    const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*,
    int32_t, int32_t, int32_t, const RopeRungs, int32_t, cudaStream_t,
    uint8_t*, int64_t, int64_t, int32_t, int32_t);

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
    const RopeRungs rungs,
    int32_t rope_interleaved,
    void* stream_ptr,
    const uint32_t* sel_entries,
    const uint32_t* sel_cnt,
    const uint2* sel_pages,
    const uint2* sel_page_win,
    int32_t sel_stride,
    int32_t sel_ratio,
    uint8_t* stage_buf,
    int64_t stage_bytes,
    int64_t stage_positions,
    int32_t stage_min_q_len,
    int32_t stage_max_kv
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    switch (q_dtype) {
        case ArenaFormat::F16:
            run_paged_prefill_int8_fp16(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, head_dim,
                max_q_len, softmax_scale, rungs,
                rope_interleaved, stream, sel_entries, sel_cnt, sel_pages, sel_page_win,
                sel_stride, sel_ratio,
                stage_buf, stage_bytes, stage_positions, stage_min_q_len, stage_max_kv);
            break;
        case ArenaFormat::BF16:
            run_paged_prefill_int8_bf16(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, head_dim,
                max_q_len, softmax_scale, rungs,
                rope_interleaved, stream, sel_entries, sel_cnt, sel_pages, sel_page_win,
                sel_stride, sel_ratio,
                stage_buf, stage_bytes, stage_positions, stage_min_q_len, stage_max_kv);
            break;
        default:
            fprintf(stderr, "run_paged_prefill_int8: unsupported q_dtype %d\n", q_dtype);
            break;
    }
}

extern "C" void run_paged_prefill_kv_stage(
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    int32_t batch_size,
    int32_t n_kv_head,
    int32_t head_dim,
    int32_t kv_dtype,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    void* stream_ptr,
    uint8_t* stage_buf,
    int64_t stage_bytes,
    int64_t stage_positions,
    int32_t stage_min_q_len,
    int32_t stage_max_kv
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    switch (kv_dtype) {
        case ArenaFormat::F16:
            run_paged_prefill_kv_stage_fp16(
                k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens, batch_size,
                n_kv_head, head_dim, rungs, rope_interleaved, stream,
                stage_buf, stage_bytes, stage_positions, stage_min_q_len, stage_max_kv);
            break;
        case ArenaFormat::BF16:
            run_paged_prefill_kv_stage_bf16(
                k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens, batch_size,
                n_kv_head, head_dim, rungs, rope_interleaved, stream,
                stage_buf, stage_bytes, stage_positions, stage_min_q_len, stage_max_kv);
            break;
        default:
            fprintf(stderr, "run_paged_prefill_kv_stage: unsupported kv_dtype %d\n", kv_dtype);
            break;
    }
}
