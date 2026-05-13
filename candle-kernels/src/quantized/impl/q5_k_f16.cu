// =============================================================================
// Q5_K + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_K (5-bit with per-block scales, K-quant)
// Y Type: F16 (half precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q5_k_f16_s1                     (single batch decode)
//     q5_k_f16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q5_k_f16_s64, q5_k_f16_s64_tc   (64-batch, smem weight tiles)
//     q5_k_f16_s128, q5_k_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_k.cuh"

constexpr int QK5_K_K128 = 128;
constexpr int QI5_K_K128 = 32;
constexpr int VDR_Q5_K_K128 = 2;

// Kernel instantiation with F16 Y vectors and K/128 format
INSTANTIATE_KERNELS(
    q5_k_f16,
    QK5_K_K128, QI5_K_K128, block_c_q5_K, VDR_Q5_K_K128,
    __half, __half
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<true, half, half, half, block_c_q5_K>(
//     const half*, const int4*, half*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q5_K_f16(const half* a, const int4* w, half* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<true, half, half, half, block_c_q5_K>(a, w, o, s, b, n, k, st, d);
// }}