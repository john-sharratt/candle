// =============================================================================
// Q2_K + Q8 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q2_K (2-bit with per-block scales)
// Y Type: Q8 (quantized activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q2_K_q8_s1                    (single batch decode)
//     q2_K_q8_s8                    (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q2_K_q8_s64, q2_K_q8_s64_tc   (64-batch, smem weight tiles)
//     q2_K_q8_s128, q2_K_q8_s128_tc (128-batch, high-smem HW)
//   L2 chunked (compute-bound):
//     q2_K_q8_s64_tc_chunked        (64-batch, L2 pipeline)
//     q2_K_q8_s128_tc_chunked       (128-batch, L2 pipeline)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q2_K.cuh"

// K/128 format parameters:
// qk = 128 elements per block
// qi = 32 (16 threads × vdr=2)
// vdr = 2 (vector dimension register)
constexpr int QK2_K_K128 = 128;
constexpr int QI2_K_K128 = 32;
constexpr int VDR_Q2_K_K128 = 2;

// Kernel instantiation with Q8_1 quantized Y vectors
INSTANTIATE_KERNELS(
    q2_K_q8,
    QK2_K_K128, QI2_K_K128, block_c_q2_K, VDR_Q2_K_K128,
    block_q8_1, __nv_bfloat16
)