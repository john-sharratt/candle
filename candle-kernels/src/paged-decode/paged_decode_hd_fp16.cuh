// =============================================================================
// paged_decode_hd_fp16.cuh — one head dim's worth of FP16 decode, per TU.
//
// The BF16 twin, `paged_decode_hd_bf16.cuh`, carries the explanation of why the
// head dims are split across translation units rather than named from one.
//
// Included by `paged_decode_fp16_hd*.cu`, each of which defines `DECODE_HD`
// before including it.
// =============================================================================

#ifndef DECODE_HD
#error "define DECODE_HD (the head dim) before including paged_decode_hd_fp16.cuh"
#endif

#include "int8_decode_kernel.cuh"

#include <cuda_fp16.h>

#define DECODE_CAT_(a, b) a##b
#define DECODE_CAT(a, b) DECODE_CAT_(a, b)

/// One head dim of the FP16 decode launcher.
///
/// Carries the union of the plain and fused-q8 argument lists so that both entry
/// points in the dispatcher reach the same instantiation: `q8_out` null is the
/// plain path writing through `o_ptr`, non-null is the B2 fused q8a128 context
/// path, as `fused_attn::launch_int8_decode_attn` documents.
extern "C" int32_t DECODE_CAT(run_paged_decode_fp16_hd, DECODE_HD)(
    const void* q_ptr,
    const uint8_t* headers_ptr,
    void* o_ptr,
    int32_t num_active_slots,
    int32_t n_q_head,
    int32_t n_kv_head,
    float softmax_scale,
    const void* k_new,
    const void* v_new,
    const float* rope_cs,
    int32_t rope_interleaved,
    void* stream_ptr,
    void* q8_out,
    const void* gate,
    int64_t gate_slot_stride
) {
    return fused_attn::launch_int8_decode_attn<__half, __half, __half, DECODE_HD>(
        (const __half*)q_ptr, headers_ptr, (__half*)o_ptr, num_active_slots, n_q_head,
        n_kv_head, softmax_scale, (const __half*)k_new, (const __half*)v_new, rope_cs,
        rope_interleaved, (cudaStream_t)stream_ptr, (uint8_t*)q8_out, (const __half*)gate,
        gate_slot_stride);
}
