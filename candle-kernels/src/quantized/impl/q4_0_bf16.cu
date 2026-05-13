// =============================================================================
// Q4_0 + BF16 KERNEL INSTANTIATION (K-TILE-MAJOR FORMAT)
// =============================================================================
// Quantization: Q4_0 (4-bit with per-block scale) in K-tile-major format
// Y Type: BF16 (bfloat16 activations)
//
// K-tile-major uses super-block parameters:
//   QK4_0_KTILE = 256 (8 GGML blocks = 16 K-tiles per super-block)
//   QI4_0_KTILE = 32  (16 threads per super-block)
//   VDR_Q4_0_KTILE = 2
// =============================================================================

#define Y_TYPE_BF16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_0.cuh"

// Kernel instantiation with BF16 Y vectors using K-tile-major parameters
INSTANTIATE_KERNELS(
    q4_0_bf16,
    QK4_0_KTILE, QI4_0_KTILE, block_c_q4_0, VDR_Q4_0_KTILE,
    __nv_bfloat16, __nv_bfloat16
)
