#include "paged_prefill_kernel.cuh"

// API — FP16 dtype
// HEAD_DIMs: 64, 96, 128, 256
extern "C" void run_paged_prefill_chunks_fp16(
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
    bool has_prefix,
    const uint32_t* rope_offsets,
    const float* rope_cs,
    int32_t rope_interleaved,
    const uint32_t* write_offset_shifts,
    int gap_fill,
    const uint32_t* col_actual_pos,
    const uint32_t* cu_kvlens,
    const uint32_t* glue_write_slice,
    const uint32_t* glue_write_in_blk,
    cudaStream_t stream
) {
    #define LAUNCH_FP16(HD, WARPS, TILE) \
        launch_paged_prefill_chunks<__half, __half, __half, HD, WARPS, TILE>( \
            q_ptr, k_ptr, v_ptr, headers_ptr, \
            cu_seqlens_q, q_lens, kv_lens, o_ptr, total_q, batch_size, \
            n_head, n_kv_head, max_blocks, softmax_scale, \
            has_prefix, rope_offsets, rope_cs, rope_interleaved, write_offset_shifts, stream, \
            (bool)gap_fill, col_actual_pos, cu_kvlens, glue_write_slice, glue_write_in_blk)
    switch (head_dim) {
        case 64:  LAUNCH_FP16(64,  4, 32); break;
        case 96:  LAUNCH_FP16(96,  4, 32); break;
        case 128: LAUNCH_FP16(128, 4, 32); break;
        case 256: LAUNCH_FP16(256, 8, 16); break;
        default: break;
    }
    #undef LAUNCH_FP16
}
