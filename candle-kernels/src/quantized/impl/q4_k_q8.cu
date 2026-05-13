// =============================================================================
// Q4_K + Q8 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q4_K (4-bit with per-block scales, K-quant) in K/128 format
// Y Type: Q8 (quantized activations)
//
// K/128 format: 128 elements per block, 16 threads per block, 8 elements per thread
// Block type: block_c_q4_K (typedef to block_c_q4_K_k128, 80 bytes)
//
// Generates batch-specialized kernels:
//   Register-only (memory-bound):
//     q4_k_q8_s1  (single batch decode)
//     q4_k_q8_s8  (octet batch, 8× weight reuse)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_k.cuh"

// K/128 format parameters:
// qk = 128 elements per block
// qi = 32 (16 threads × vdr=2)
// vdr = 2 (vector dimension register)
constexpr int QK4_K_K128 = 128;
constexpr int QI4_K_K128 = 32;
constexpr int VDR_Q4_K_K128 = 2;

// Kernel instantiation with Q8 Y vectors
INSTANTIATE_KERNELS(
    q4_k_q8,
    QK4_K_K128, QI4_K_K128, block_c_q4_K, VDR_Q4_K_K128,
    block_q8_1, __nv_bfloat16
)
