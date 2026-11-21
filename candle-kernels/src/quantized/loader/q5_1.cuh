#pragma once

// =============================================================================
// Q5_1 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 5-bit weights (4-bit low + 1-bit high)
// Q5_1 has separate d and m scales (not derived like Q5_0)
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "q5_1/k_tile_loader.cuh"
// =============================================================================
// GEMX DEQUANT TRAITS - Q5_1 (5-bit asymmetric: value = d * q5 + m)
// =============================================================================
#include "gemx_dequant.cuh"

#include "q5_1/mma_dequant.cuh"
