// =============================================================================
// Q5_K + Q8 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_K (5-bit with per-block scales, K-quant)
// Y Type: Q8 (quantized activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q5_k_q8_s1                    (single batch decode)
//     q5_k_q8_s8                    (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q5_k_q8_s64, q5_k_q8_s64_tc   (64-batch, smem weight tiles)
//     q5_k_q8_s128, q5_k_q8_s128_tc (128-batch, high-smem HW)
//   L2 chunked (compute-bound):
//     q5_k_q8_s64_tc_chunked        (64-batch, L2 pipeline)
//     q5_k_q8_s128_tc_chunked       (128-batch, L2 pipeline)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_k.cuh"

constexpr int QK5_K_K128 = 128;
constexpr int QI5_K_K128 = 32;
constexpr int VDR_Q5_K_K128 = 2;

// Kernel instantiation with Q8 Y vectors and K/128 format
INSTANTIATE_KERNELS(
    q5_k_q8,
    QK5_K_K128, QI5_K_K128, block_c_q5_K, VDR_Q5_K_K128,
    block_q8_1, __nv_bfloat16
)
