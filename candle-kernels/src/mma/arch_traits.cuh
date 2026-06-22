#pragma once
// =============================================================================
// arch_traits.cuh — per-architecture compile-time MMA shape constants.
//
// The fused attention kernel uses INT8 MMA throughout. The MMA tile shape
// differs across Ampere / Ada / Blackwell:
//
//   sm_86 (Ampere 3000-series): m16n8k16
//   sm_89 (Ada 4000-series):    m16n8k32
//   sm_120 (Blackwell):         m16n8k32 (forward-compat path)
//
// HEAD_DIM=128, N_PALETTE=4 -> palette covers 32 dims, which equals MMA_K on
// Ada/Blackwell (one MMA per palette) and is double MMA_K on Ampere (two MMAs
// per palette, sum partials inside palette).
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace fused_attn {

template<int SM_VERSION>
struct ArchTraits;

template<>
struct ArchTraits<89> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 16;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<>
struct ArchTraits<86> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 18;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<>
struct ArchTraits<120> {
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 32;

    static constexpr int A_REGS_PER_THREAD = 1;
    static constexpr int B_REGS_PER_THREAD = 1;
    static constexpr int C_REGS_PER_THREAD = 4;

    template<int HEAD_DIM, int N_PALETTE>
    static constexpr int mmas_per_palette() {
        return (HEAD_DIM / N_PALETTE) / MMA_K;
    }

    static constexpr int MMA_LATENCY_CYCLES = 11;
    static constexpr int MMA_ISSUE_INTERVAL = 4;
};

template<int SM_VERSION>
constexpr bool is_supported_arch() {
    return SM_VERSION == 86 || SM_VERSION == 89 || SM_VERSION == 120;
}

} // namespace fused_attn
