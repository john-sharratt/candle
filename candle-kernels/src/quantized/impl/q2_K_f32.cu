// =============================================================================
// Q2_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q2_K (2-bit K-quant with per-block scales)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

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

// Kernel instantiation with F32 Y vectors and F32 output
INSTANTIATE_KERNELS(
    q2_K_f32,
    QK2_K_K128, QI2_K_K128, block_c_q2_K, VDR_Q2_K_K128,
    float, float
)