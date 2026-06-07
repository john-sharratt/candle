#pragma once
// =============================================================================
// model_descriptor.cuh — compile-time shape parameterization.
//
// One model = one binary specialization. Shape tuple drives codegen; multiple
// models with identical shape compile to identical kernels.
// =============================================================================

#include "arch_traits.cuh"

namespace fused_attn {

enum class RopeStyle : int {
    Full    = 0,
    Partial = 1,
};

template<
    int        HEAD_DIM_,
    int        N_Q_HEADS_,
    int        N_KV_HEADS_,
    int        D_MODEL_,
    int        N_PALETTE_,
    RopeStyle  ROPE_STYLE_,
    bool       USE_QK_NORM_,
    bool       USE_SLIDING_WINDOW_,
    bool       ROPE_INTERLEAVED_>
struct ModelDescriptor {
    static constexpr int       HEAD_DIM           = HEAD_DIM_;
    static constexpr int       N_Q_HEADS          = N_Q_HEADS_;
    static constexpr int       N_KV_HEADS         = N_KV_HEADS_;
    static constexpr int       D_MODEL            = D_MODEL_;
    static constexpr int       N_PALETTE          = N_PALETTE_;
    static constexpr RopeStyle ROPE_STYLE         = ROPE_STYLE_;
    static constexpr bool      USE_QK_NORM        = USE_QK_NORM_;
    static constexpr bool      USE_SLIDING_WINDOW = USE_SLIDING_WINDOW_;
    static constexpr bool      ROPE_INTERLEAVED   = ROPE_INTERLEAVED_;

    static constexpr int GQA_GROUP        = N_Q_HEADS / N_KV_HEADS;
    static constexpr int Q_OUTPUT_DIM     = N_Q_HEADS  * HEAD_DIM;
    static constexpr int K_OUTPUT_DIM     = N_KV_HEADS * HEAD_DIM;
    static constexpr int V_OUTPUT_DIM     = N_KV_HEADS * HEAD_DIM;
    static constexpr int TOTAL_OUTPUT_DIM = Q_OUTPUT_DIM + K_OUTPUT_DIM + V_OUTPUT_DIM;
    static constexpr int DIMS_PER_PALETTE = HEAD_DIM / N_PALETTE;
    static constexpr int MIN_BATCH_FOR_KERNEL = (16 + GQA_GROUP - 1) / GQA_GROUP;

    static_assert(HEAD_DIM == 128,
        "v1 only supports HEAD_DIM=128.");
    static_assert(N_PALETTE == 4,
        "v1 only supports N_PALETTE=4 due to deferred-scaling alignment.");
    static_assert(N_Q_HEADS % N_KV_HEADS == 0,
        "GQA requires n_q_heads divisible by n_kv_heads.");
    static_assert(D_MODEL % 32 == 0,
        "D_MODEL must be divisible by MMA K-dim (32).");
    static_assert(DIMS_PER_PALETTE == 32,
        "Deferred scaling requires DIMS_PER_PALETTE == 32 = MMA_K on Ada.");
};

} // namespace fused_attn
