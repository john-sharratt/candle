// =============================================================================
// Q5_0 + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_0 (5-bit with per-block scale)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_0.cuh"

// Kernel instantiation with F32 Y vectors and F32 output
// K-TILE-MAJOR: Uses QK5_0_KTILE=256, QI5_0_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q5_0_f32,
    QK5_0_KTILE, QI5_0_KTILE, block_c_q5_0, VDR_Q5_0_KTILE,
    float, float
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<false, half, float, float, block_c_q5_0>(
//     const float*, const int4*, float*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q5_0_f32(const float* a, const int4* w, float* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<false, half, float, float, block_c_q5_0>(a, w, o, s, b, n, k, st, d);
// }}