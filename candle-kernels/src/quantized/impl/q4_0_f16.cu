// =============================================================================
// Q4_0 + F16 KERNEL INSTANTIATION (K-TILE-MAJOR FORMAT)
// =============================================================================
// Quantization: Q4_0 (4-bit with per-block scale) in K-tile-major format
// Y Type: F16 (half precision activations)
//
// K-tile-major uses super-block parameters:
//   QK4_0_KTILE = 256 (8 GGML blocks = 16 K-tiles per super-block)
//   QI4_0_KTILE = 32  (16 threads per super-block)
//   VDR_Q4_0_KTILE = 2
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q4_0_f16_s1                    (single batch decode)
//     q4_0_f16_s8                    (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q4_0_f16_s64, q4_0_f16_s64_tc   (64-batch, smem weight tiles)
//     q4_0_f16_s128, q4_0_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_0.cuh"

// Kernel instantiation with F16 Y vectors using K-tile-major parameters
INSTANTIATE_KERNELS(
    q4_0_f16,
    QK4_0_KTILE, QI4_0_KTILE, block_c_q4_0, VDR_Q4_0_KTILE,
    __half, __half
)
