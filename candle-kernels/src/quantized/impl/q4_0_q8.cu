// =============================================================================
// Q4_0 + Q8 KERNEL INSTANTIATION (K-TILE-MAJOR FORMAT)
// =============================================================================
// Quantization: Q4_0 (4-bit with per-block scale) in K-tile-major format
// Y Type: Q8 (quantized activations)
//
// K-tile-major uses super-block parameters:
//   QK4_0_KTILE = 256 (8 GGML blocks = 16 K-tiles per super-block)
//   QI4_0_KTILE = 32  (16 threads per super-block)
//   VDR_Q4_0_KTILE = 2
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_0.cuh"

// Kernel instantiation with Q8 Y vectors using K-tile-major parameters
INSTANTIATE_KERNELS(
    q4_0_q8,
    QK4_0_KTILE, QI4_0_KTILE, block_c_q4_0, VDR_Q4_0_KTILE,
    block_q8_1, __nv_bfloat16
)
