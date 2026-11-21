// Batched Penalty and Sampling Kernel for efficient token generation
// This kernel processes all sequences in a batch with a single kernel launch,
// eliminating the per-sequence kernel launch overhead.
// 
// Key features:
// - Single kernel launch for N sequences (was N launches)
// - O(1) bitset lookup for recent tokens
// - O(n) radix select for top-k (avoids full vocab sort)
// - Truly branchless penalty application
// - Temperature <= 0 triggers argmax (greedy) path
// - Supports FP32, FP16, BF16, and FP8 E4M3 input logits
// - float4 vectorized loads for FP32 (4x memory bandwidth)

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>  // For std::is_same_v
#include "philox.cuh"
#include "../fast_exp.cuh"

namespace batched_sampling {

#include "batched_sampling/sampling_types.cuh"

#include "batched_sampling/penalty_helpers.cuh"

#include "batched_sampling/vocab_passes.cuh"

#include "batched_sampling/argmax_topk.cuh"

#include "batched_sampling/topk_sampling.cuh"

#include "batched_sampling/stencil_sampling.cuh"

#include "batched_sampling/main_kernel.cuh"

#include "batched_sampling/launch_wrappers.cuh"

} // namespace batched_sampling
