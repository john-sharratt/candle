// =============================================================================
// Q3_K + BF16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q3_K (3-bit with per-block scales)
// Y Type: BF16 (bfloat16 activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q3_k_bf16_s1                      (single batch decode)
//     q3_k_bf16_s8                      (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q3_k_bf16_s64, q3_k_bf16_s64_tc   (64-batch, smem weight tiles)
//     q3_k_bf16_s128, q3_k_bf16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_BF16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q3_k.cuh"

// K/128 format parameters:
// qk = 128 elements per block
// qi = 32 (16 threads × vdr=2)
// vdr = 2 (vector dimension register)
constexpr int QK3_K_K128 = 128;
constexpr int QI3_K_K128 = 32;
constexpr int VDR_Q3_K_K128 = 2;

// Kernel instantiation with BF16 Y vectors
INSTANTIATE_KERNELS(
    q3_k_bf16,
    QK3_K_K128, QI3_K_K128, block_c_q3_K, VDR_Q3_K_K128,
    __nv_bfloat16, __nv_bfloat16
)