// =============================================================================
// launch.cu — shape-keyed launchers + single C-callable dispatch entry.
// =============================================================================

#include "attn_fused_v1.cuh"
#include "model_descriptor.cuh"
#include "v2_compat_dispatch.cuh"
#include "../paged-decode/paged_decode_kernel.cuh"  // commit_decode_write_len_kernel

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace fused_attn {

// -----------------------------------------------------------------------------
// Concrete shape launchers — one per (shape × feature flags × architecture).
// These are explicit instantiations; calling each one compiles a dedicated
// kernel binary specialized for the shape.
// -----------------------------------------------------------------------------

#define DEFINE_SHAPE_LAUNCHER(                                                       \
    HEAD_DIM, NQH, NKVH, DMODEL,                                                     \
    RSTYLE_TAG, RILVD, QKN, SW,                                                      \
    SM, NAME)                                                                        \
    template<typename Q_T, typename O>                                               \
    cudaError_t NAME(                                                                \
        const Q_T*     activations,                                                  \
        const uint8_t* w_qkv_q4,                                                     \
        const void*    w_qkv_scales,                                                 \
        const uint8_t* headers_ptr,                                                  \
        O*             out,                                                          \
        int            num_active_slots,                                             \
        int            n_q_head,                                                     \
        int            n_kv_head,                                                    \
        float          softmax_scale,                                                \
        const float*   rope_cs,                                                      \
        int            sliding_window_size,                                          \
        cudaStream_t   stream = nullptr                                              \
    ) {                                                                              \
        using Cfg = ModelDescriptor<                                                 \
            HEAD_DIM, NQH, NKVH, DMODEL, /*N_PALETTE=*/4,                            \
            RopeStyle::RSTYLE_TAG, /*USE_QK_NORM=*/QKN,                              \
            /*USE_SLIDING_WINDOW=*/SW, /*ROPE_INTERLEAVED=*/RILVD>;                  \
        return launch_fused_qkv_attn<Q_T, O, Cfg, SM>(                               \
            activations, w_qkv_q4, w_qkv_scales,                                     \
            headers_ptr, out,                                                        \
            num_active_slots, n_q_head, n_kv_head,                                   \
            softmax_scale, rope_cs,                                                  \
            sliding_window_size, stream);                                            \
    }

// sm_89 (Ada — primary dev target)
DEFINE_SHAPE_LAUNCHER(128, 32,  4, 2048, Full, false, false, false, 89,
                       launch_fused_attn_h128_q32_kv4_d2048_sm89)
DEFINE_SHAPE_LAUNCHER(128, 32,  8, 4096, Full, false, false, false, 89,
                       launch_fused_attn_h128_q32_kv8_d4096_sm89)
DEFINE_SHAPE_LAUNCHER(128, 32,  8, 4096, Full, false, false, true,  89,
                       launch_fused_attn_h128_q32_kv8_d4096_swin_sm89)
DEFINE_SHAPE_LAUNCHER(128, 24,  8, 3072, Full, false, false, false, 89,
                       launch_fused_attn_h128_q24_kv8_d3072_sm89)
DEFINE_SHAPE_LAUNCHER(128, 16,  8, 2048, Full, false, false, false, 89,
                       launch_fused_attn_h128_q16_kv8_d2048_sm89)

// sm_86 (Ampere)
DEFINE_SHAPE_LAUNCHER(128, 32,  4, 2048, Full, false, false, false, 86,
                       launch_fused_attn_h128_q32_kv4_d2048_sm86)
DEFINE_SHAPE_LAUNCHER(128, 32,  8, 4096, Full, false, false, false, 86,
                       launch_fused_attn_h128_q32_kv8_d4096_sm86)

// sm_120 (Blackwell)
DEFINE_SHAPE_LAUNCHER(128, 32,  4, 2048, Full, false, false, false, 120,
                       launch_fused_attn_h128_q32_kv4_d2048_sm120)
DEFINE_SHAPE_LAUNCHER(128, 32,  8, 4096, Full, false, false, false, 120,
                       launch_fused_attn_h128_q32_kv8_d4096_sm120)

#undef DEFINE_SHAPE_LAUNCHER

} // namespace fused_attn

// -----------------------------------------------------------------------------
// C dispatch entry point — Rust calls this directly.
// -----------------------------------------------------------------------------

extern "C" int fused_attn_v1_dispatch(
    // Shape selectors
    int            head_dim,
    int            n_q_head,
    int            n_kv_head,
    int            d_model,
    int            rope_interleaved,
    int            rope_style,         // 0=Full, 1=Partial
    int            use_qk_norm,
    int            use_sliding_window,
    int            sm_version,
    int            q_dtype_tag,        // 0=F16, 1=BF16
    int            o_dtype_tag,        // 0=F16, 1=BF16

    // Runtime args
    const void*    activations,
    const uint8_t* w_qkv_q4,
    const void*    w_qkv_scales,
    const uint8_t* headers_ptr,
    void*          out,
    int            num_active_slots,
    float          softmax_scale,
    const float*   rope_cs,
    int            sliding_window_size,
    void*          stream_ptr
) {
    using namespace fused_attn;
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    auto shape_matches = [&](int hd, int nq, int nkv, int dm, bool ilv,
                              int rs, bool qkn, bool sw) {
        return head_dim == hd && n_q_head == nq && n_kv_head == nkv
            && d_model == dm && (rope_interleaved != 0) == ilv
            && rope_style == rs && (use_qk_norm != 0) == qkn
            && (use_sliding_window != 0) == sw;
    };

    #define DISPATCH_DTYPES(LAUNCHER)                                              \
        do {                                                                       \
            if (q_dtype_tag == 0 && o_dtype_tag == 0) {                            \
                return (int)LAUNCHER<__half, __half>(                              \
                    (const __half*)activations,                                    \
                    w_qkv_q4, w_qkv_scales, headers_ptr,                           \
                    (__half*)out,                                                  \
                    num_active_slots, n_q_head, n_kv_head,                         \
                    softmax_scale, rope_cs, sliding_window_size, stream);          \
            }                                                                      \
            if (q_dtype_tag == 1 && o_dtype_tag == 1) {                            \
                return (int)LAUNCHER<__nv_bfloat16, __nv_bfloat16>(                \
                    (const __nv_bfloat16*)activations,                             \
                    w_qkv_q4, w_qkv_scales, headers_ptr,                           \
                    (__nv_bfloat16*)out,                                           \
                    num_active_slots, n_q_head, n_kv_head,                         \
                    softmax_scale, rope_cs, sliding_window_size, stream);          \
            }                                                                      \
            return (int)cudaErrorNotSupported;                                     \
        } while (0)

    if (sm_version == 89) {
        if (shape_matches(128, 32,  4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm89);
        if (shape_matches(128, 32,  8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm89);
        if (shape_matches(128, 32,  8, 4096, false, 0, false, true))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_swin_sm89);
        if (shape_matches(128, 24,  8, 3072, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q24_kv8_d3072_sm89);
        if (shape_matches(128, 16,  8, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q16_kv8_d2048_sm89);
    }
    if (sm_version == 86) {
        if (shape_matches(128, 32,  4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm86);
        if (shape_matches(128, 32,  8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm86);
    }
    if (sm_version == 120) {
        if (shape_matches(128, 32,  4, 2048, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv4_d2048_sm120);
        if (shape_matches(128, 32,  8, 4096, false, 0, false, false))
            DISPATCH_DTYPES(launch_fused_attn_h128_q32_kv8_d4096_sm120);
    }

    #undef DISPATCH_DTYPES

    return (int)cudaErrorNotSupported;
}

// =============================================================================
// v2-API-compatible dispatch entry — pre-projected Q/K/V.
//
// Signature mirrors run_paged_decode_fp16/bf16 exactly so it can be plugged
// into `candle::CustomOp1::cuda_fwd` for `PagedDecode` directly.
//
// Currently a Phase-1 passthrough: forwards to v2's launch_paged_decode_attn.
// Replace internals incrementally with INT8 MMA paths.
// =============================================================================

extern "C" int fused_attn_v1_v2_compat_dispatch(
    int            q_dtype_tag,    // 0=F16, 1=BF16
    int            head_dim,
    const void*    q_ptr,
    const uint8_t* headers_ptr,
    void*          o_ptr,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const void*    k_new,
    const void*    v_new,
    const float*   rope_cs,
    int            rope_interleaved,
    void*          stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    using namespace fused_attn;

    #define LAUNCH_HD(HD, QT)                                                        \
        do {                                                                          \
            return (int)launch_v2_compat<QT, QT, QT, HD>(                            \
                (const QT*)q_ptr, headers_ptr, (QT*)o_ptr,                           \
                num_active_slots, n_q_head, n_kv_head, softmax_scale,                \
                (const QT*)k_new, (const QT*)v_new, rope_cs,                         \
                rope_interleaved, stream);                                           \
        } while (0)

    // HEAD_DIM=256 is intentionally NOT instantiated: the INT8 decode kernel's
    // shared-memory footprint at HD=256 (~78 KB static) exceeds the 48 KB ptxas
    // static cap. HD=256 is not a target-model head dim (Qwen3/Llama are all
    // hd128), so callers fall back to the v2 paged-decode kernel for it.
    if (q_dtype_tag == 0) {  // FP16
        switch (head_dim) {
            case 64:  LAUNCH_HD(64,  __half);
            case 96:  LAUNCH_HD(96,  __half);
            case 128: LAUNCH_HD(128, __half);
            default:  return (int)cudaErrorNotSupported;
        }
    } else if (q_dtype_tag == 1) {  // BF16
        switch (head_dim) {
            case 64:  LAUNCH_HD(64,  __nv_bfloat16);
            case 96:  LAUNCH_HD(96,  __nv_bfloat16);
            case 128: LAUNCH_HD(128, __nv_bfloat16);
            default:  return (int)cudaErrorNotSupported;
        }
    }
    #undef LAUNCH_HD
    return (int)cudaErrorNotSupported;
}
