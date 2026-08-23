// =============================================================================
// paged_decode_api_bf16.cu — default decode dispatch (BF16).
//
// The INT8 decode kernel (split-KV / warp-stripe / batched-M) is the production
// path for head_dim 64/96/128/256. head_dim 256 runs its wide (hpg>8) path
// single-stage so the tiles fit the 48 KiB static shared-memory cap; the stripe
// and batched-M paths are unchanged.
// =============================================================================

#include "int8_decode_kernel.cuh"

#include <cuda_bf16.h>

// Returns 0 on success, nonzero when the launch needed the split-KV partial
// pool and its allocation failed (VRAM exhausted) — nothing was launched.
extern "C" int32_t run_paged_decode_bf16(
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
        return fused_attn::launch_int8_decode_attn<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, HD>( \
            (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr, \
            num_active_slots, n_q_head, n_kv_head, softmax_scale, \
            (const __nv_bfloat16*)k_new, (const __nv_bfloat16*)v_new, rope_cs, rope_interleaved, stream)
    switch (head_dim) {
        case 64:  LAUNCH_INT8(64);
        case 96:  LAUNCH_INT8(96);
        case 128: LAUNCH_INT8(128);
        case 256: LAUNCH_INT8(256);
        default: return 0; // rust dispatch bails before reaching an unsupported width
    }
    #undef LAUNCH_INT8
}

// B2: decode with fused q8a128 context output (feeds o_proj directly, no standalone
// quantize). head_dim 128 and 256 — a combine block covers head_dim/128 whole
// q8a128 tiles. `gate` (nullable, bf16) folds the output gate sigmoid(g) ⊙ ctx
// into the same pass (gated lineages, head_dim 256); a slot's n_q_head×head_dim
// gate values are contiguous and consecutive slots are `gate_slot_stride`
// elements apart, so the gate may be a strided view of the fused [q|gate]
// projection (pass 0 for a fully contiguous gate).
// Returns 0 on success, nonzero when the partial pool (which every q8 emit
// requires) could not be allocated — nothing was launched.
extern "C" int32_t run_paged_decode_bf16_q8(
    const void* q_ptr,
    const uint8_t* headers_ptr,
    void* q8_out,
    const void* gate,
    int64_t gate_slot_stride,
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
    #define LAUNCH_Q8(HD)                                                                  \
        return fused_attn::launch_int8_decode_attn<__nv_bfloat16, __nv_bfloat16,           \
                                                   __nv_bfloat16, HD>(                     \
            (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)nullptr,             \
            num_active_slots, n_q_head, n_kv_head, softmax_scale,                          \
            (const __nv_bfloat16*)k_new, (const __nv_bfloat16*)v_new, rope_cs,             \
            rope_interleaved, stream, (uint8_t*)q8_out, (const __nv_bfloat16*)gate,        \
            gate_slot_stride)
    switch (head_dim) {
        case 128: LAUNCH_Q8(128);
        case 256: LAUNCH_Q8(256);
        default: return 0; // rust dispatch bails before reaching an unsupported width
    }
    #undef LAUNCH_Q8
}

// =============================================================================
// Regression-test entry for the int8 m16n8k32 MMA fragment loaders.
//
// Computes C[16][8] = A[16][32] · B[8][32]^T from row-major int8 A, B (so
// C[m][n] = sum_k A[m][k] * B[n][k]) using the exact loaders the decode QK dot
// uses — load_a_frag_m16k32 / load_b_frag_n8k32 / mma_int8_m16n8k32. A wrong
// per-thread fragment decomposition (the bug that silently corrupted every int8
// QK dot) makes C disagree with a trivial CPU reference, so the int8_mma test in
// candle-transformers' prefill_utils catches it instantly. Single warp; not on
// any production path.
// =============================================================================
namespace {
__global__ void mma_int8_m16n8k32_test_kernel(
    const int8_t* __restrict__ A,   // 16 x 32 row-major
    const int8_t* __restrict__ B,   // 8  x 32 row-major (n x k)
    int32_t* __restrict__ C         // 16 x 8 row-major
) {
    __shared__ alignas(16) int8_t sA[16 * 32];
    __shared__ alignas(16) int8_t sB[8 * 32];
    int t = (int)threadIdx.x;
    for (int i = t; i < 16 * 32; i += 32) sA[i] = A[i];
    for (int i = t; i < 8 * 32; i += 32) sB[i] = B[i];
    __syncthreads();
    uint32_t a_frag[4];
    fused_attn::load_a_frag_m16k32(a_frag, sA, 32, t);
    uint32_t b_frag[2];
    fused_attn::load_b_frag_n8k32(b_frag, sB, 32, t);
    int32_t c[4] = {0, 0, 0, 0};
    fused_attn::mma_int8_m16n8k32(c, a_frag, b_frag, c);
    // m16n8 C fragment: c0=C[m,n], c1=C[m,n+1], c2=C[m+8,n], c3=C[m+8,n+1]
    // with m = lane>>2, n = (lane&3)*2.
    int m = t >> 2;
    int n = (t & 3) * 2;
    C[m * 8 + n]           = c[0];
    C[m * 8 + n + 1]       = c[1];
    C[(m + 8) * 8 + n]     = c[2];
    C[(m + 8) * 8 + n + 1] = c[3];
}
} // namespace

extern "C" void mma_int8_m16n8k32_test(
    const int8_t* A, const int8_t* B, int32_t* C, void* stream_ptr
) {
    mma_int8_m16n8k32_test_kernel<<<1, 32, 0, (cudaStream_t)stream_ptr>>>(A, B, C);
}
