// =============================================================================
// Q_AWQ_G64 + BF16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q_AWQ_G64 (4-bit AWQ with group size 64)
// Y Type: BF16 (bfloat16 activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q_awq_g64_bf16_s1                     (single batch decode)
//     q_awq_g64_bf16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q_awq_g64_bf16_s64, q_awq_g64_bf16_s64_tc   (64-batch, smem weight tiles)
//     q_awq_g64_bf16_s128, q_awq_g64_bf16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_BF16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q_awq_g64.cuh"

// Kernel instantiation with BF16 Y vectors
// K/128 format: Uses QK_Q_AWQ_G64_KTILE=128, QI_Q_AWQ_G64_KTILE=16, VDR=1 → 16 threads/K/128 block
INSTANTIATE_KERNELS(
    q_awq_g64_bf16,
    QK_Q_AWQ_G64_KTILE, QI_Q_AWQ_G64_KTILE, block_c_q_awq_g64, VDR_Q_AWQ_G64_KTILE,
    __nv_bfloat16, __nv_bfloat16
)
