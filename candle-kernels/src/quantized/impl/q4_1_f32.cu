// =============================================================================
// Q4_1 + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q4_1 (4-bit with per-block scale and min)
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_1.cuh"

// Kernel instantiation with F32 Y vectors and F32 output
// K-TILE-MAJOR: Uses QK4_1_KTILE=256, QI4_1_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q4_1_f32,
    QK4_1_KTILE, QI4_1_KTILE, block_c_q4_1, VDR_Q4_1_KTILE,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q4_1 (4-bit weights × q8a128 int8
// activations, deferred affine {d, m} fold, F32 output). See grouped_tc_int8 in kernel.cuh.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q4_1_int8_f32,
    QK4_1_KTILE, QI4_1_KTILE, block_c_q4_1, VDR_Q4_1_KTILE,
    float
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<false, half, float, float, block_c_q4_1>(
//     const float*, const int4*, float*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q4_1_f32(const float* a, const int4* w, float* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<false, half, float, float, block_c_q4_1>(a, w, o, s, b, n, k, st, d);
// }}