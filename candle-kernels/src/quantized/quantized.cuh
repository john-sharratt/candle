#pragma once

// =============================================================================
// QUANTIZED KERNEL MASTER INCLUDE
// =============================================================================
// Single entry point for all quantized kernel infrastructure.
// Enforces proper dependency order for compilation.
//
// Include order (respecting dependencies):
//   1. types.cuh        - Compile-time type traits (no dependencies)
//   2. math.cuh         - Math utilities (depends on types)
//   3. reduce.cuh       - Reduction/accumulation (depends on types, math)
//   4. helpers.cuh      - Low-level utilities (no dependencies)
//   5. loaders.cuh      - Loader trait dispatch (depends on helpers, math, reduce, and includes all loader/*.cuh)
//   6. process_tile.cuh - Tile processing (depends on math, loaders)
//   7. pipeline.cuh     - Double-buffered pipeline (depends on math, loaders)
//   8. kernel.cuh - Main kernel (depends on all above)
//
// Usage: #include "quantized/quantized.cuh"
//
// =============================================================================

// Stage 1: Type System Foundation
#include "types.cuh"       // accumulator_type<>, tile_k_blocks_v<>, can_double_buffer_v<>

// Stage 2: Math Utilities (includes reductions)
#include "math.cuh"        // Type conversions, operators, warp_reduce_sum_t<>, accumulate(), acc_to_float()

// Stage 3: Low-level Utilities
#include "helpers.cuh"     // get_int_from_*(), ggml_cuda_dp4a(), WARP_SIZE

// Stage 5: Loader Framework (includes all 10 loader specializations)
#include "loaders.cuh"     // vec_dot_loader_for<> trait + loader/generic.cuh
                           //   + loader/q4_0.cuh, q4_1.cuh, q5_0.cuh, q5_1.cuh, q8_0.cuh
                           //   + loader/q2_K.cuh, q3_K.cuh, q4_K.cuh, q5_K.cuh, q6_K.cuh

// Stage 6: Tile Processing
#include "process_tile.cuh" // process_full_tile<>(), process_partial_tile<>()

// Stage 7: Main Kernel Template
#include "kernel.cuh" // quantized_gemv<>() - the main kernel function
