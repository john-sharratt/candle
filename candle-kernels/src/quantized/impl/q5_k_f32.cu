// =============================================================================
// Q5_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_K (5-bit K-quant with per-block scales)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_K.cuh"

constexpr int QK5_K_K128 = 128;
constexpr int QI5_K_K128 = 32;
constexpr int VDR_Q5_K_K128 = 2;

// Kernel instantiation with F32 Y vectors and K/128 format
INSTANTIATE_KERNELS(
    q5_k_f32,
    QK5_K_K128, QI5_K_K128, block_c_q5_K, VDR_Q5_K_K128,
    float, float
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<true, half, float, float, block_c_q5_K>(
//     const float*, const int4*, float*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q5_K_f32(const float* a, const int4* w, float* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<true, half, float, float, block_c_q5_K>(a, w, o, s, b, n, k, st, d);
// }}