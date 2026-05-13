// =============================================================================
// Q4_1 + Q8 KERNEL INSTANTIATION (K/128 FORMAT)
// =============================================================================
// Quantization: Q4_1 (4-bit with per-block scale and min) in K/128 format
// Y Type: Q8 (quantized activations)
//
// K/128 format: 128 elements per block, 16 threads per block, 8 elements per thread
// Block type: block_c_q4_1 (typedef to block_c_q4_1_k128, 80 bytes)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_1.cuh"

// K/128 format parameters
constexpr int QK4_1_K128 = 128;
constexpr int QI4_1_K128 = 32;
constexpr int VDR_Q4_1_K128 = 2;

// Kernel instantiation with Q8 Y vectors using K/128 parameters
INSTANTIATE_KERNELS(
    q4_1_q8,
    QK4_1_K128, QI4_1_K128, block_c_q4_1, VDR_Q4_1_K128,
    block_q8_1, __nv_bfloat16
)
