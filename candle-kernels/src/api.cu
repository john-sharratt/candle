// =============================================================================
// CANDLE-KERNELS UNIFIED API
// =============================================================================
// This file provides extern "C" wrapper functions for all candle-kernels.
// Rust calls these wrappers via FFI, and the wrappers launch the CUDA kernels.
//
// Organization:
//   1. Simple kernels (affine, binary, unary, etc.) - direct wrappers
//   2. Quantized kernels - dispatch by quant type
//   3. Flash-attention kernels - already have their own api.cu files
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cstddef>

#include "api/kernel_decls.cuh"

// =============================================================================
// WRAPPER FUNCTIONS
// =============================================================================

extern "C" {

#include "api/op_wrappers.cuh"

#include "api/compare_inplace_unary.cuh"

#include "api/typed_wrappers.cuh"

#include "api/dispatchers.cuh"

}  // extern "C"
