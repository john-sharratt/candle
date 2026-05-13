#pragma once

// =============================================================================
// COMMON KERNEL INFRASTRUCTURE (included after Y_TYPE definition)
// =============================================================================
// This file must be included AFTER defining Y_TYPE_* to provide all necessary
// types, macros, and the main kernel function template.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>
#include <algorithm>

// =============================================================================
// GEMX LAYOUT CONSTANTS
// =============================================================================
// For [K/block, N] layout (GEMX), we need num_rows to compute indices.
// This is set via cudaMemcpyToSymbol before kernel launch.
//
// Block index:  block_idx = row + kbx * d_num_rows
// Scale index:  scale_idx = (kbx * scales_per_block + offset) * d_num_rows + row
extern __device__ __constant__ int d_num_rows;

// =============================================================================
// GEMX SCALE PERMUTATION HELPERS
// =============================================================================
// Reorders scales within 64-element chunks to match GEMX's tensor core
// thread access pattern. This is an 8x8 transpose within each chunk.
//
// gemx_permute_64:   Used at extraction (write-time) to arrange scales
// gemx_unpermute_64: Used at load (read-time) to recover logical index
//
// Note: 8x8 transpose is self-inverse, so both are the same operation.
// Two names exist for code clarity / self-documentation.

__device__ __forceinline__ int gemx_permute_64(int linear_idx) {
    int chunk_idx = linear_idx >> 6;              // linear_idx / 64
    int idx_in_chunk = linear_idx & 63;           // linear_idx % 64
    int permuted = ((idx_in_chunk & 7) << 3) | (idx_in_chunk >> 3);  // 8x8 transpose
    return (chunk_idx << 6) | permuted;
}

// Alias for readability - same operation since 8x8 transpose is self-inverse
__device__ __forceinline__ int gemx_unpermute_64(int linear_idx) {
    return gemx_permute_64(linear_idx);
}

// =============================================================================
// CONFIGURATION MACROS
// =============================================================================

#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_ASSUME(x)

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

// Aliases for K-quant block sizes (all use QK_K)
#define QK2_K QK_K
#define QK3_K QK_K
#define QK4_K QK_K
#define QK5_K QK_K
#define QK6_K QK_K

#undef GGML_CUDA_F16
#define GGML_CUDA_DMMV_X 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

// Launch bounds for different kernel profiles:
// VSMALL: _s1-_s3 - 4 blocks/SM, ~60 reg budget (high occupancy)
// SMALL:  _s4-_s8 - 2 blocks/SM, ~128 reg budget
// ITER:   s2_iter kernels - 4 blocks/SM, ~60 regs (high occupancy)
// TC16:   tc16 kernels - 8 blocks/SM, ~56-64 regs (high occupancy tensor core)
// TC32:   tc32 kernels - 5 blocks/SM, ~80 regs (larger batch tile, more registers)
#define LAUNCH_BOUNDS_VSMALL __launch_bounds__(128, 4)
#define LAUNCH_BOUNDS_SMALL __launch_bounds__(128, 2)
#define LAUNCH_BOUNDS_ITER __launch_bounds__(128, 4)
#define LAUNCH_BOUNDS_VEC __launch_bounds__(128, 1)
#define LAUNCH_BOUNDS_TC16 __launch_bounds__(128, 10)  // 10 blocks/SM target
#define LAUNCH_BOUNDS_TC32 __launch_bounds__(128, 8)  // 8 blocks/SM, ~15% faster FP8

// =============================================================================
// BLOCK TYPE DEFINITIONS
// =============================================================================
// All block types (dtype and quant) are defined in the central blocks.cuh file.
// This include brings in: block_f32, block_f16, block_bf16, block_fp8_e4m3,
// block_q4_0/1, block_q5_0/1, block_q8_0/1, block_q2_K through block_q8_K,
// block_q_awq, block_q_awq_g64, and all associated QK/QR/QI/VDR constants.

#include "../../blocks.cuh"

// =============================================================================
// GGML MACROS (for compatibility)
// =============================================================================

#define GGML_CUDA_DMMV_X 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

// =============================================================================
// VEC_DOT VECTOR DECODE RATE (VDR) CONSTANTS
// =============================================================================
// VDR = vec dot ratio, how many contiguous integers each thread processes
// Separate constants for MMVQ (mul_mat_vec_q) and MMQ (mul_mat_q) paths

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ  8

#define VDR_Q8_1_Q8_1_MMVQ 2
#define VDR_Q8_1_Q8_1_MMQ  8

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// Q6_K K/64 uses VDR=2 (4 threads per K/64 block, 16 elements per thread)
#define VDR_Q6_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMQ  8

// AWQ (4-bit asymmetric) - similar to Q4_1 with scale and zero
#define VDR_Q_AWQ_Q8_1_MMVQ 2
#define VDR_Q_AWQ_Q8_1_MMQ  4

#define VDR_Q_AWQ_G64_Q8_1_MMVQ 2
#define VDR_Q_AWQ_G64_Q8_1_MMQ  4

// =============================================================================
// WARP SIZE CONSTANT
// =============================================================================
#define WARP_SIZE 32

// =============================================================================
// FUNCTION POINTER TYPES FOR VEC_DOT OPERATIONS
// =============================================================================
typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

// =============================================================================
// DEQUANTIZATION OUTPUT SIZE INFORMATION
// =============================================================================
// Helper template to determine how many output elements a dequant call produces
// and how many input bytes it consumes

// Default: single block consumed (in bytes), 32 elements produced (e.g., Q8_1)
template <typename block_t>
struct dequant_sizes {
    static constexpr int input_stride = sizeof(block_t);  // Bytes consumed per call
    static constexpr int output_count = 32;               // Elements produced per call
};

// Specializations for DUAL-BLOCK types (consume 2 blocks in bytes, produce 64 elements)
template <> struct dequant_sizes<block_q4_0> { 
    static constexpr int input_stride = sizeof(block_q4_0) * 2; 
    static constexpr int output_count = 64; 
};
template <> struct dequant_sizes<block_q4_1> { 
    static constexpr int input_stride = sizeof(block_q4_1) * 2; 
    static constexpr int output_count = 64; 
};
template <> struct dequant_sizes<block_q5_0> { 
    static constexpr int input_stride = sizeof(block_q5_0) * 2; 
    static constexpr int output_count = 64; 
};
template <> struct dequant_sizes<block_q5_1> { 
    static constexpr int input_stride = sizeof(block_q5_1) * 2; 
    static constexpr int output_count = 64; 
};
template <> struct dequant_sizes<block_q8_0> { 
    static constexpr int input_stride = sizeof(block_q8_0) * 2; 
    static constexpr int output_count = 64; 
};

// Specializations for K-quant types (consume 1 block in bytes, produce 256 elements)
template <> struct dequant_sizes<block_q2_K> { 
    static constexpr int input_stride = sizeof(block_q2_K); 
    static constexpr int output_count = QK_K; 
};
template <> struct dequant_sizes<block_q3_K> { 
    static constexpr int input_stride = sizeof(block_q3_K); 
    static constexpr int output_count = QK_K; 
};
template <> struct dequant_sizes<block_q4_K> { 
    static constexpr int input_stride = sizeof(block_q4_K); 
    static constexpr int output_count = QK_K; 
};
template <> struct dequant_sizes<block_q5_K> { 
    static constexpr int input_stride = sizeof(block_q5_K); 
    static constexpr int output_count = QK_K; 
};
template <> struct dequant_sizes<block_q6_K> { 
    static constexpr int input_stride = sizeof(block_q6_K); 
    static constexpr int output_count = QK_K; 
};
template <> struct dequant_sizes<block_q8_K> { 
    static constexpr int input_stride = sizeof(block_q8_K); 
    static constexpr int output_count = QK_K; 
};

// Include K/128 compact types (must come before specializations use them)
#include "../block_compact.cuh"

// Specializations for K/128 compact types (K-tile-major with embedded scales)
// These produce 128 elements per block (half of GGML's 256)
template <> struct dequant_sizes<block_c_q2_K> { 
    static constexpr int input_stride = sizeof(block_c_q2_K);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q3_K> { 
    static constexpr int input_stride = sizeof(block_c_q3_K);  // 96 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q4_K> { 
    static constexpr int input_stride = sizeof(block_c_q4_K);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q5_K> { 
    static constexpr int input_stride = sizeof(block_c_q5_K);  // 112 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q6_K> { 
    static constexpr int input_stride = sizeof(block_c_q6_K);  // 128 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q8_K> { 
    static constexpr int input_stride = sizeof(block_c_q8_K);  // 160 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};

// Specializations for simple quant K/128 types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
// These are typedefs to block_c_qX_Y_k128 types
template <> struct dequant_sizes<block_c_q4_0> { 
    static constexpr int input_stride = sizeof(block_c_q4_0);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q4_1> { 
    static constexpr int input_stride = sizeof(block_c_q4_1);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q5_0> { 
    static constexpr int input_stride = sizeof(block_c_q5_0);  // 112 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q5_1> { 
    static constexpr int input_stride = sizeof(block_c_q5_1);  // 112 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q8_0> { 
    static constexpr int input_stride = sizeof(block_c_q8_0);  // 144 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};

// AWQ K/128 types
template <> struct dequant_sizes<block_c_q_awq> { 
    static constexpr int input_stride = sizeof(block_c_q_awq);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};
template <> struct dequant_sizes<block_c_q_awq_g64> { 
    static constexpr int input_stride = sizeof(block_c_q_awq_g64);  // 80 bytes
    static constexpr int output_count = 128;  // K/128 format: 128 elements per block
};

// Helper variable templates (C++14+)
template <typename block_t>
constexpr int dequant_input_stride_v = dequant_sizes<block_t>::input_stride;

template <typename block_t>
constexpr int dequant_output_count_v = dequant_sizes<block_t>::output_count;

// =============================================================================
// INCLUDE ALL INFRASTRUCTURE
// =============================================================================
// Always define Y_TYPE_Q8 so loaders can compile their Q8_1 dot paths
#define Y_TYPE_Q8

#include "../types.cuh"
#include "../math.cuh"
#include "../reduce.cuh"
#include "../helpers.cuh"
#include "../kernel.cuh"
#include "../process_tile.cuh"

