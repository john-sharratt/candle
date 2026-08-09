// =============================================================================
// Q_AWQ + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q_AWQ (4-bit AWQ with group size 128)
// Y Type: F32 (single precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q_awq_f32_s1                     (single batch decode)
//     q_awq_f32_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q_awq_f32_s64, q_awq_f32_s64_tc   (64-batch, smem weight tiles)
//     q_awq_f32_s128, q_awq_f32_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q_awq.cuh"

// Kernel instantiation with F32 Y vectors
// K/128 format: Uses QK_Q_AWQ_KTILE=128, QI_Q_AWQ_KTILE=16, VDR=1 → 16 threads/K/128 block
INSTANTIATE_KERNELS(
    q_awq_f32,
    QK_Q_AWQ_KTILE, QI_Q_AWQ_KTILE, block_c_q_awq, VDR_Q_AWQ_KTILE,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for AWQ (4-bit affine weights, one
// scale/zero per 128-block, F32 output). See grouped_tc_int8.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q_awq_int8_f32,
    QK_Q_AWQ_KTILE, QI_Q_AWQ_KTILE, block_c_q_awq, VDR_Q_AWQ_KTILE,
    float
)
INSTANTIATE_KERNEL_DENSE_INT8_ALL(
    q_awq_int8,
    QK_Q_AWQ_KTILE, QI_Q_AWQ_KTILE, block_c_q_awq, VDR_Q_AWQ_KTILE
)
