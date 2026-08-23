#include "paged_prefill_int8_kernel.cuh"

// INT8 prefix-attention prefill — BF16 Q/K/V/O.
// HEAD_DIMs: 64, 128, 256 (HEAD_DIM % 64 == 0 for in-thread RoPE pairing).
// 256 runs at 2 blocks/SM on a 4-way output-dim split; see i8_dim_split.
extern "C" void run_paged_prefill_int8_bf16(
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
    const uint32_t* rope_offsets,
    const float* rope_cs,
    int32_t rope_interleaved,
    cudaStream_t stream
) {
    using prefill_int8::launch_paged_prefill_int8;
    switch (head_dim) {
        case 64:
            launch_paged_prefill_int8<__nv_bfloat16, 64>(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, max_q_len,
                softmax_scale, rope_offsets, rope_cs, rope_interleaved, stream);
            break;
        case 128:
            launch_paged_prefill_int8<__nv_bfloat16, 128>(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, max_q_len,
                softmax_scale, rope_offsets, rope_cs, rope_interleaved, stream);
            break;
        case 256:
            launch_paged_prefill_int8<__nv_bfloat16, 256>(
                q_ptr, k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                o_ptr, total_q, batch_size, n_head, n_kv_head, max_q_len,
                softmax_scale, rope_offsets, rope_cs, rope_interleaved, stream);
            break;
        default:
            fprintf(stderr, "run_paged_prefill_int8_bf16: unsupported head_dim %d\n", head_dim);
            break;
    }
}
