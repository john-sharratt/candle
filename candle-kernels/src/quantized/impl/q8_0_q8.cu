// =============================================================================
// Q8_0 + Q8 KERNEL INSTANTIATION (K/128 FORMAT)
// =============================================================================
// Quantization: Q8_0 (8-bit with per-block scale) in K/128 format
// Y Type: Q8 (quantized activations)
//
// K/128 format: 128 elements per block, 16 threads per block, 8 elements per thread
// Block type: block_c_q8_0 (typedef to block_c_q8_0_k128, 144 bytes)
// =============================================================================

#define Y_TYPE_Q8

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_0.cuh"

// K/128 format parameters
constexpr int QK8_0_K128 = 128;
constexpr int QI8_0_K128 = 32;
constexpr int VDR_Q8_0_K128 = 2;

// Kernel instantiation with Q8 Y vectors using K/128 parameters
INSTANTIATE_KERNELS(
    q8_0_q8,
    QK8_0_K128, QI8_0_K128, block_c_q8_0, VDR_Q8_0_K128,
    block_q8_1, __nv_bfloat16
)
