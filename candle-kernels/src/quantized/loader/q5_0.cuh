#pragma once

// =============================================================================
// Q5_0 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 5-bit weights (4-bit low + 1-bit high)
// - qs: 32 bits = 8 × 4-bit low nibbles
// - qh: 8 bits = 8 × 1-bit high bits
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "q5_0/k_tile_loader.cuh"
// =============================================================================
// GEMX DEQUANT TRAITS - Q5_0 (5-bit symmetric: value = d * (q5 - 16))
// =============================================================================
#include "gemx_dequant.cuh"

#include "q5_0/mma_dequant.cuh"
