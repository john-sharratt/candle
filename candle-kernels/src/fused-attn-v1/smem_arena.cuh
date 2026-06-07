#pragma once
// =============================================================================
// smem_arena.cuh — phase-based shared-memory union view.
//
// A single __shared__ SmemArena<Cfg> serves all four phases via a union of
// two phase-specific views.
//
// Note on W_qkv staging size:
//   The full TOTAL_OUTPUT_DIM (e.g. 5120 for Llama-3.2-3B) does not fit in
//   smem if staged at K-chunk granularity. We tile across the N-dim using
//   W_QKV_TILE_N (default 128) so per-stage staging stays bounded. Consumer
//   warps step through N tiles inside the K-chunk loop.
// =============================================================================

#include "model_descriptor.cuh"
#include "arch_traits.cuh"

namespace fused_attn {

namespace tile {
    static constexpr int TILE_N             = 32;  // KV tokens per attention tile
    static constexpr int N_PIPELINE_STAGES  = 3;
    static constexpr int N_W_STAGING_STAGES = 2;
    static constexpr int W_QKV_TILE_N       = 128; // output-dim tile for W_qkv staging
}

template<typename Cfg>
struct Phase12View {
    // Activation vector — INT8 + per-32-block scale.
    int8_t activations_int8 [Cfg::D_MODEL];
    float  activations_scales[Cfg::D_MODEL / 32];

    // W_qkv Q4_0 source — one K-chunk (32 K-elements) x W_QKV_TILE_N output dims
    // = 32 * W_QKV_TILE_N / 2 bytes (Q4 packs 2 nibbles per byte).
    static constexpr int W_Q4_BYTES_PER_STAGE  = 32 * tile::W_QKV_TILE_N / 2;
    static constexpr int W_INT8_BYTES_PER_STAGE = 32 * tile::W_QKV_TILE_N;

    uint8_t w_q4_src[tile::N_W_STAGING_STAGES][W_Q4_BYTES_PER_STAGE];
    int8_t  w_staging_int8 [tile::N_W_STAGING_STAGES][W_INT8_BYTES_PER_STAGE];
    float   w_staging_scales[tile::N_W_STAGING_STAGES][tile::W_QKV_TILE_N / 32];

    // K_new / V_new FP32 intermediate, written by consumers, consumed by
    // dequant role's phase 3.
    float k_new_fp32[Cfg::N_KV_HEADS * Cfg::HEAD_DIM];
    float v_new_fp32[Cfg::N_KV_HEADS * Cfg::HEAD_DIM];
};

template<typename Cfg>
struct Phase4View {
    static constexpr int Q4_BYTES_PER_TOKEN = Cfg::HEAD_DIM / 2;

    // Q4 packed source — loader writes here, dequant warps consume.
    uint8_t smem_q_K[tile::N_PIPELINE_STAGES][tile::TILE_N][Q4_BYTES_PER_TOKEN];
    uint8_t smem_q_V[tile::N_PIPELINE_STAGES][tile::TILE_N][Q4_BYTES_PER_TOKEN];

    // INT8 buffers for MMA consumption.
    //   K is k-major: smem_int8_K[stage][dim][token]
    //   V is mn-major: smem_int8_V[stage][token][dim]
    int8_t smem_int8_K[tile::N_PIPELINE_STAGES][Cfg::HEAD_DIM][tile::TILE_N];
    int8_t smem_int8_V[tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::HEAD_DIM];

    // Scale tables (per-tile, per-token, per-palette).
    float smem_scale_K_pre [tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::N_PALETTE];
    float smem_scale_K_post[tile::N_PIPELINE_STAGES][tile::TILE_N][Cfg::N_PALETTE];
    float smem_scale_V     [tile::N_PIPELINE_STAGES][tile::TILE_N];

    // Per-token RoPE positions for K (W2 reads these during dequant + RoPE).
    uint32_t k_rope_positions[tile::N_PIPELINE_STAGES][tile::TILE_N];
};

template<typename Cfg>
union SmemArena {
    Phase12View<Cfg> phase12;
    Phase4View<Cfg>  phase4;

    __device__ Phase12View<Cfg>& as_phase12() { return phase12; }
    __device__ Phase4View<Cfg>&  as_phase4()  { return phase4;  }
};

template<typename Cfg>
constexpr bool smem_arena_fits_default() {
    return sizeof(SmemArena<Cfg>) <= 48 * 1024;
}

} // namespace fused_attn
