#include <stdint.h>
#include <cassert>
#include "../arena_table.cuh"  // For CHUNK_SIZE constant and ArenaFormat namespace

// ============================================================================
// Paged Prefill API - Unified Dispatcher
//
// Delegates to per-dtype dispatchers (defined in separate translation units)
// which each contain a switch(head_dim) over the supported head dimensions.
// ============================================================================

// Per-dtype dispatcher signature (head_dim dispatched internally)
using paged_prefill_dtype_fn_t = void (*) (
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, bool, const uint32_t*, const float*, int32_t, const uint32_t*,
    cudaStream_t);

extern "C" void run_paged_prefill_chunks_fp16(
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, bool, const uint32_t*, const float*, int32_t, const uint32_t*,
    cudaStream_t);

extern "C" void run_paged_prefill_chunks_bf16(
    const void*, const void*, const void*, const uint8_t*,
    const uint32_t*, const uint32_t*, const uint32_t*, void*,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
    float, bool, const uint32_t*, const float*, int32_t, const uint32_t*,
    cudaStream_t);

// ============================================================================
// Main dispatcher - q_dtype selects per-dtype function
// ============================================================================
extern "C" void run_paged_prefill_chunks(
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
    int32_t max_blocks,
    float softmax_scale,
    int32_t q_dtype,  // Q/K/V/O dtype: 0=F32, 1=F16, 2=BF16
    int has_prefix,
    const uint32_t* rope_offsets,
    const float* rope_cs,
    int32_t rope_interleaved,
    const uint32_t* write_offset_shifts,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    paged_prefill_dtype_fn_t fn = nullptr;
    switch (q_dtype) {
        case ArenaFormat::F16:  fn = &run_paged_prefill_chunks_fp16; break;
        case ArenaFormat::BF16: fn = &run_paged_prefill_chunks_bf16; break;
    }

    if (fn != nullptr) {
        fn(q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q,
           q_lens, kv_lens, o_ptr, total_q, batch_size, n_head, n_kv_head,
           head_dim, max_blocks, softmax_scale, (bool)has_prefix,
           rope_offsets, rope_cs, rope_interleaved, write_offset_shifts, stream);
    }
}
