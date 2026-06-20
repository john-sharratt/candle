// =============================================================================
// Q3_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q3_K (3-bit K-quant with per-block scales)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q3_K.cuh"

// K/128 format parameters:
// qk = 128 elements per block
// qi = 32 (16 threads × vdr=2)
// vdr = 2 (vector dimension register)
constexpr int QK3_K_K128 = 128;
constexpr int QI3_K_K128 = 32;
constexpr int VDR_Q3_K_K128 = 2;

// Kernel instantiation with F32 Y vectors and F32 output
INSTANTIATE_KERNELS(
    q3_k_f32,
    QK3_K_K128, QI3_K_K128, block_c_q3_K, VDR_Q3_K_K128,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q3_K (3-bit symmetric weights, per-16
// scales via split 2-MMA, centered int8, F32 output). See grouped_tc_int8.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q3_k_int8_f32,
    QK3_K_K128, QI3_K_K128, block_c_q3_K, VDR_Q3_K_K128,
    float
)