// =============================================================================
// Q8_1 + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_1 (8-bit with per-block scale and sum)
// Y Type: F16 (half precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q8_1_f16_s1                     (single batch decode)
//     q8_1_f16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q8_1_f16_s64, q8_1_f16_s64_tc   (64-batch, smem weight tiles)
//     q8_1_f16_s128, q8_1_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_1.cuh"

// Kernel instantiation with F16 Y vectors
// K-TILE-MAJOR: Uses QK8_1_KTILE=128, QI8_1_KTILE=16, VDR=1 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q8_1_f16,
    QK8_1_KTILE, QI8_1_KTILE, block_c_q8_1, VDR_Q8_1_KTILE,
    __half, __half
)
