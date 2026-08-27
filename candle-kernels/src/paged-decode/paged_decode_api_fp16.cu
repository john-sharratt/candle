// =============================================================================
// paged_decode_api_fp16.cu — default decode dispatch (FP16).
//
// The INT8 decode kernel (split-KV / warp-stripe / batched-M) is the production
// path for head_dim 64/96/128/256. head_dim 256 runs its wide (hpg>8) path
// single-stage so the tiles fit the 48 KiB static shared-memory cap; the stripe
// and batched-M paths are unchanged.
// =============================================================================

// **No kernel header here.** Unlike the BF16 twin, this file has no test kernel
// of its own, so with the head dims in their own TUs (`paged_decode_fp16_hd*.cu`)
// nothing left in it needs one — see `paged_decode_hd_bf16.cuh`.
#include <cstdint>

extern "C" {
int32_t run_paged_decode_fp16_hd64(const void*, const uint8_t*, void*, int32_t, int32_t,
                                   int32_t, float, const void*, const void*, const float*,
                                   int32_t, void*, void*, const void*, int64_t);
int32_t run_paged_decode_fp16_hd96(const void*, const uint8_t*, void*, int32_t, int32_t,
                                   int32_t, float, const void*, const void*, const float*,
                                   int32_t, void*, void*, const void*, int64_t);
int32_t run_paged_decode_fp16_hd128(const void*, const uint8_t*, void*, int32_t, int32_t,
                                    int32_t, float, const void*, const void*, const float*,
                                    int32_t, void*, void*, const void*, int64_t);
int32_t run_paged_decode_fp16_hd256(const void*, const uint8_t*, void*, int32_t, int32_t,
                                    int32_t, float, const void*, const void*, const float*,
                                    int32_t, void*, void*, const void*, int64_t);
}

// Returns 0 on success, nonzero when the launch needed the split-KV partial
// pool and its allocation failed (VRAM exhausted) — nothing was launched.
extern "C" int32_t run_paged_decode_fp16(
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
    // Null q8/gate: the plain path, writing through `o_ptr`.
    #define LAUNCH_INT8(HD) \
        return run_paged_decode_fp16_hd##HD( \
            q_ptr, headers_ptr, o_ptr, num_active_slots, n_q_head, n_kv_head, \
            softmax_scale, k_new, v_new, rope_cs, rope_interleaved, stream_ptr, \
            nullptr, nullptr, 0)
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
// q8a128 tiles. `gate` (nullable, fp16) folds the output gate sigmoid(g) ⊙ ctx
// into the same pass (gated lineages, head_dim 256); a slot's n_q_head×head_dim
// gate values are contiguous and consecutive slots are `gate_slot_stride`
// elements apart, so the gate may be a strided view of the fused [q|gate]
// projection (pass 0 for a fully contiguous gate).
// Returns 0 on success, nonzero when the partial pool (which every q8 emit
// requires) could not be allocated — nothing was launched.
extern "C" int32_t run_paged_decode_fp16_q8(
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
    // Non-null q8_out: the combine kernel is the only emitter, so `out` is null.
    #define LAUNCH_Q8(HD)                                                                  \
        return run_paged_decode_fp16_hd##HD(                                               \
            q_ptr, headers_ptr, nullptr, num_active_slots, n_q_head, n_kv_head,            \
            softmax_scale, k_new, v_new, rope_cs, rope_interleaved, stream_ptr,            \
            q8_out, gate, gate_slot_stride)
    switch (head_dim) {
        case 128: LAUNCH_Q8(128);
        case 256: LAUNCH_Q8(256);
        default: return 0; // rust dispatch bails before reaching an unsupported width
    }
    #undef LAUNCH_Q8
}
