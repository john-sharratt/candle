// =============================================================================
// Q8_K + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_K (8-bit K-quant with 256-element super-blocks)
// Y Type: F16 (half precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q8_k_f16_s1                     (single batch decode)
//     q8_k_f16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q8_k_f16_s64, q8_k_f16_s64_tc   (64-batch, smem weight tiles)
//     q8_k_f16_s128, q8_k_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_K.cuh"

// Kernel instantiation with F16 Y vectors
// K/128 format: Uses QK8_K_KTILE=128, QI8_K_KTILE=16, VDR=1 → 16 threads/K/128 block
INSTANTIATE_KERNELS(
    q8_k_f16,
    QK8_K_KTILE, QI8_K_KTILE, block_c_q8_K, VDR_Q8_K_KTILE,
    __half, __half
)
