// =============================================================================
// Q4_0 + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q4_0 (4-bit with per-block scale)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_0.cuh"

// Kernel instantiation with F32 Y vectors and F32 output
// Using K-tile-major parameters for GEMX format
INSTANTIATE_KERNELS(
    q4_0_f32,
    QK4_0_KTILE, QI4_0_KTILE, block_c_q4_0, VDR_Q4_0_KTILE,
    float, float
)
