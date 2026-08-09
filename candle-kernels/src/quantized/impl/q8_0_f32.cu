// =============================================================================
// Q8_0 + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_0 (8-bit with per-block scale)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_0.cuh"

// Kernel instantiation with F32 Y vectors and F32 output
// K-TILE-MAJOR: Uses QK8_0_KTILE=256, QI8_0_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q8_0_f32,
    QK8_0_KTILE, QI8_0_KTILE, block_c_q8_0, VDR_Q8_0_KTILE,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q8_0 (raw signed int8 weights ×
// q8a128 int8 activations, no min, F32 output). See grouped_tc_int8 in kernel.cuh.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q8_0_int8_f32,
    QK8_0_KTILE, QI8_0_KTILE, block_c_q8_0, VDR_Q8_0_KTILE,
    float
)
INSTANTIATE_KERNEL_DENSE_INT8_ALL(
    q8_0_int8,
    QK8_0_KTILE, QI8_0_KTILE, block_c_q8_0, VDR_Q8_0_KTILE
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<false, half, float, float, block_c_q8_0>(
//     const float*, const int4*, float*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q8_0_f32(const float* a, const int4* w, float* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<false, half, float, float, block_c_q8_0>(a, w, o, s, b, n, k, st, d);
// }}