#pragma once

#include "helpers.cuh"
#include "math.cuh"
#include "reduce.cuh"
#include "block_compact.cuh"

// =============================================================================
// QUANTIZED LOADER TYPE TRAIT - Dispatches to specialized loader implementations
// =============================================================================
// This header provides the primary template for vec_dot_loader_for<block_q_t, vdr, act_t>.
// Each quantization format has its own specialized loader in the loader/ subfolder.
//
// Usage pattern:
//   using loader_type = typename vec_dot_loader_for<block_q4_0, 1, float>::type;
//   loader_type loader;
//   loader.load_x(bq4_0, iqs);           // Load quantized block
//   float result = loader.dot_y(y_data); // Compute dot product with Y
//
// Primary template - all specializations defined in loader/*.cuh
// act_t determines the internal scale format:
//   float        → float2
//   half         → half2
//   __nv_bfloat16 → __nv_bfloat162
//   __nv_fp8_e4m3 → half2
template <typename block_q_t, int vdr = 1, typename act_t = float> struct vec_dot_loader_for;

// =============================================================================
// LOADER PARTS TRAIT - Number of load_part/dequant iterations for each format
// =============================================================================
// Different formats split their work into different numbers of parts:
//   - NUM_PARTS=1: Simple load_part<0>() + dot_y() (non-K-quants, all K-tile-major)
//   - NUM_PARTS>1: Legacy multi-part loaders (being deprecated)
template <typename block_q_t> struct loader_num_parts { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q4_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q4_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q5_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q5_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q8_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q8_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q8_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q2_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q3_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q4_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q5_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q6_K> { static constexpr int value = 1; };

// Compact block types (when kernel is instantiated with block_c_* directly)
// Values must match the corresponding original block types above
template <> struct loader_num_parts<block_c_q4_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q4_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q5_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q5_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q8_0> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q8_1> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q8_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q2_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q3_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q4_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q5_K> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q6_K> { static constexpr int value = 1; };

// AWQ types
template <> struct loader_num_parts<block_q_awq> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_q_awq_g64> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q_awq> { static constexpr int value = 1; };
template <> struct loader_num_parts<block_c_q_awq_g64> { static constexpr int value = 1; };

template <typename block_q_t>
constexpr int loader_num_parts_v = loader_num_parts<block_q_t>::value;

// NOTE: Loader specializations are NOT included here to avoid recompilation cascade.
// Each impl/*.cu file includes its specific loader directly (e.g., loader/q6_K.cuh).
// This ensures editing one loader only recompiles kernels for that format.
//
// Only generic.cuh is included here since it provides the base class used by all loaders.
#include "loader/generic.cuh"

// =============================================================================
// TYPE TRAITS FOR COMPACTED BLOCKS
// =============================================================================
// Maps original block type to compacted type. Defined here after both original
// types (from helpers.cuh -> impl/common.cuh) and compacted types (from 
// block_compact.cuh) are available.

template <typename block_t> struct block_compact;

template <> struct block_compact<block_q4_0> { using type = block_c_q4_0; };
template <> struct block_compact<block_q4_1> { using type = block_c_q4_1; };
template <> struct block_compact<block_q5_0> { using type = block_c_q5_0; };
template <> struct block_compact<block_q5_1> { using type = block_c_q5_1; };
template <> struct block_compact<block_q8_0> { using type = block_c_q8_0; };
template <> struct block_compact<block_q8_1> { using type = block_c_q8_1; };
template <> struct block_compact<block_q8_K> { using type = block_c_q8_K; };
template <> struct block_compact<block_q2_K> { using type = block_c_q2_K; };
template <> struct block_compact<block_q3_K> { using type = block_c_q3_K; };
template <> struct block_compact<block_q4_K> { using type = block_c_q4_K; };
template <> struct block_compact<block_q5_K> { using type = block_c_q5_K; };
template <> struct block_compact<block_q6_K> { using type = block_c_q6_K; };

// Identity mappings for compact block types (when kernel is instantiated with block_c_* directly)
template <> struct block_compact<block_c_q4_0> { using type = block_c_q4_0; };
template <> struct block_compact<block_c_q4_1> { using type = block_c_q4_1; };
template <> struct block_compact<block_c_q5_0> { using type = block_c_q5_0; };
template <> struct block_compact<block_c_q5_1> { using type = block_c_q5_1; };
template <> struct block_compact<block_c_q8_0> { using type = block_c_q8_0; };
template <> struct block_compact<block_c_q8_1> { using type = block_c_q8_1; };
template <> struct block_compact<block_c_q8_K> { using type = block_c_q8_K; };
template <> struct block_compact<block_c_q2_K> { using type = block_c_q2_K; };
template <> struct block_compact<block_c_q3_K> { using type = block_c_q3_K; };
template <> struct block_compact<block_c_q4_K> { using type = block_c_q4_K; };
template <> struct block_compact<block_c_q4_KO> { using type = block_c_q4_KO_k1024; };
template <> struct block_compact<block_c_q5_K> { using type = block_c_q5_K; };
template <> struct block_compact<block_c_q5_KO> { using type = block_c_q5_KO_k1024; };
template <> struct block_compact<block_c_q6_K> { using type = block_c_q6_K; };
template <> struct block_compact<block_c_q6_KO> { using type = block_c_q6_KO_k1024; };
template <> struct block_compact<block_c_q8_KO> { using type = block_c_q8_KO_k1024; };
template <> struct block_compact<block_c_mxfp4> { using type = block_c_mxfp4_k1024; };

// AWQ mappings
template <> struct block_compact<block_q_awq> { using type = block_c_q_awq; };
template <> struct block_compact<block_q_awq_g64> { using type = block_c_q_awq_g64; };
template <> struct block_compact<block_c_q_awq> { using type = block_c_q_awq; };
template <> struct block_compact<block_c_q_awq_g64> { using type = block_c_q_awq_g64; };

template <typename block_t>
using block_compact_t = typename block_compact<block_t>::type;
