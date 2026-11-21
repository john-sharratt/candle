// =============================================================================
// QUANTIZED MATMUL DISPATCHER WITH GREEDY BATCH DECOMPOSITION
// =============================================================================
// Provides a single extern "C" entry point (run_quantized_matmul) that dispatches
// to the appropriate typed kernel based on qtype and ytype parameters. 
//
// qtype: 0-9 (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
// ytype: 0-2 (F16, BF16, F32)
//   Note: F32 (ytype=2) only supported for Q4_K currently. Used for validation.
//
// Tensor core usage is determined automatically from SM version:
//   - SM80+ (Ampere/Ada): TC enabled for F16/BF16/F32
//
// Dispatch paths (SM80+):
//   - TC16 path (batch 1-31): tc16_0 / tc16_r kernels with internal tiling
//   - TC32 path (batch 32+):  tc32_0 / tc32_r kernels with greedy decomposition
//
// Tensor cores are used for ALL batch sizes >= 1, including batch 1-2.
// Benchmarking found the TC kernels faster than the CUDA-core GEMV (s1..s8)
// kernels in every measured case — even single-token decode — so there is no
// CUDA-core fast path on TC-capable hardware. The s1..s8 GEMV kernels remain
// only as the fallback for pre-SM80 GPUs without tensor cores.
//
// At small batch a single weight's TC grid under-fills the SMs. For MoE the
// grouped path (run_grouped_quantized_matmul) fixes this by running all experts
// in one launch so their tiles together fill the machine; the per-weight path
// here just accepts the lower occupancy.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <atomic>
#include <cstdio>
#include <cstring>

// Dispatch table for deterministic kernel selection and reporting
#include "block_compact.cuh"  // qtype_to_matmul_kernel_index + QTYPE_* constants
#include "dispatch_table.cuh"
#include "matmul_status.cuh"  // QMM_* launcher status codes
#include "moe_live.cuh"       // the grouped launch's workers over a live expert table
#include "splitk_segs.cuh"    // the split-K and narrow launches' weight segments
#include "narrow_layout.cuh"  // the narrow launch's geometry and shared memory
#include "pdl.cuh"            // programmatic dependent launch for the weight GEMMs that wait

#include "dispatcher/launch_error_flush.cuh"

// =============================================================================
// CACHED DEVICE PROPERTIES (via shared device_caps.cuh)
// =============================================================================
// Populated once per device via get_device_caps(), then reused everywhere.
// This avoids repeated cudaGetDeviceProperties() calls which are expensive.
#include "device_caps.cuh"

#include "dispatcher/grid_config.cuh"

#include "dispatcher/kernel_tables.cuh"


#include "dispatcher/dense_matmul.cuh"

#include "dispatcher/grouped_moe.cuh"

#include "dispatcher/dispatch_info.cuh"
