// =============================================================================
// Q8_1 + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_1 (8-bit with per-block scale and sum)
// Y Type: F32 (single precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q8_1_f32_s1                     (single batch decode)
//     q8_1_f32_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q8_1_f32_s64, q8_1_f32_s64_tc   (64-batch, smem weight tiles)
//     q8_1_f32_s128, q8_1_f32_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_1.cuh"

// Kernel instantiation with F32 Y vectors
// K-TILE-MAJOR: Uses QK8_1_KTILE=128, QI8_1_KTILE=16, VDR=1 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q8_1_f32,
    QK8_1_KTILE, QI8_1_KTILE, block_c_q8_1, VDR_Q8_1_KTILE,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q8_1 (8-bit symmetric weights, per-32
// scale, F32 output). See grouped_tc_int8.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q8_1_int8_f32,
    QK8_1_KTILE, QI8_1_KTILE, block_c_q8_1, VDR_Q8_1_KTILE,
    float
)
INSTANTIATE_KERNEL_DENSE_INT8_ALL(
    q8_1_int8,
    QK8_1_KTILE, QI8_1_KTILE, block_c_q8_1, VDR_Q8_1_KTILE
)
