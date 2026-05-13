// =============================================================================
// Q6_K + Q8 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q6_K (6-bit with per-block scales, K-quant)
// Y Type: Q8 (quantized activations)
// Block type: block_c_q6_K (typedef to block_c_q6_K_k128, 128 bytes)
// Elements per block: 128 (16 threads × 8 elements)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q6_k_q8_s1                    (single batch decode)
//     q6_k_q8_s8                    (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q6_k_q8_s64, q6_k_q8_s64_tc   (64-batch, smem weight tiles)
//     q6_k_q8_s128, q6_k_q8_s128_tc (128-batch, high-smem HW)
//   L2 chunked (compute-bound):
//     q6_k_q8_s64_tc_chunked        (64-batch, L2 pipeline)
//     q6_k_q8_s128_tc_chunked       (128-batch, L2 pipeline)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q6_k.cuh"

// K/128 format: 128 elements per block, 32 ints of data
constexpr int QK6_K_K128 = 128;
constexpr int QI6_K_K128 = 32;
// VDR=2 gives 16 threads per quant block (128 / (4 * 2) = 16)
constexpr int VDR_Q6_K_K128 = 2;

// Kernel instantiation with Q8 Y vectors
// K/128 format requires vdr=2 (16 threads per K/128 block)
INSTANTIATE_KERNELS(
    q6_k_q8,
    QK6_K_K128, QI6_K_K128, block_c_q6_K, VDR_Q6_K_K128,
    block_q8_1, __nv_bfloat16
)
