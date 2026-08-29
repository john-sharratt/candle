// =============================================================================
// paged_decode_hd_bf16.cuh — one head dim's worth of BF16 decode, per TU.
//
// # Why the head dims are compiled separately
//
// `launch_int8_decode_attn<…, HEAD_DIM>` expands to a whole dispatch tree: two
// warp widths, two rope modes, and eight heads-per-group instantiations of the
// stripe or batched-M kernel under each. Naming four head dims from one
// translation unit put four of those trees in front of a single nvcc process,
// which compiles them serially — `--threads` parallelises the *gencode* arches
// within one kernel, not the kernels within one file. One 6 KB source file took
// over ten minutes while the build's per-file parallelism (capped at 16 jobs)
// had nothing to do.
//
// So each head dim gets its own TU, and the `extern "C"` dispatcher in
// `paged_decode_api_bf16.cu` switches between the shims those TUs define. The
// generated code is identical — the same kernels with the same template
// arguments — it is only distributed so the jobs can run at once. It also makes
// incremental rebuilds proportional: editing something that only affects
// head_dim 128 stops recompiling 64, 96 and 256.
//
// # The duplication costs an archive, not a binary
//
// Each TU carries its own copy of the shared device helpers, so
// `libpaged_decode.a` grows — measured at 17.7 MB → 41.8 MB on the current
// baseline, which looks alarming until you measure the thing that actually
// consumes disk. The linker runs `/OPT:REF,ICF`, keeps only referenced objects
// and folds the identical copies: the `candle_core` test binary is **230 MB
// either way**, byte for byte.
//
// That distinction is the whole decision. This split was reverted once on the
// assumption that archive growth propagates to every binary linking it — 43 of
// them, so ~1 GB per build generation. It does not. The cost is one 24 MB
// archive in `target/*/build`, paid once.
//
// Measure the binary, not the archive, before undoing this.
//
// Included by `paged_decode_bf16_hd*.cu`, each of which defines `DECODE_HD`
// before including it.
// =============================================================================

#ifndef DECODE_HD
#error "define DECODE_HD (the head dim) before including paged_decode_hd_bf16.cuh"
#endif

#include "int8_decode_kernel.cuh"

#include <cuda_bf16.h>

#define DECODE_CAT_(a, b) a##b
#define DECODE_CAT(a, b) DECODE_CAT_(a, b)

/// One head dim of the BF16 decode launcher.
///
/// Carries the union of the plain and fused-q8 argument lists so that both entry
/// points in the dispatcher reach the same instantiation: `q8_out` null is the
/// plain path writing through `o_ptr`, non-null is the B2 fused q8a128 context
/// path, exactly as `fused_attn::launch_int8_decode_attn` documents.
extern "C" int32_t DECODE_CAT(run_paged_decode_bf16_hd, DECODE_HD)(
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
    return fused_attn::launch_int8_decode_attn<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                                               DECODE_HD>(
        (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr, num_active_slots,
        n_q_head, n_kv_head, softmax_scale, (const __nv_bfloat16*)k_new,
        (const __nv_bfloat16*)v_new, rope_cs, rope_interleaved, (cudaStream_t)stream_ptr,
        (uint8_t*)q8_out, (const __nv_bfloat16*)gate, gate_slot_stride);
}
