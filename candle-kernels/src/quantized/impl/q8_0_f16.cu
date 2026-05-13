// =============================================================================
// Q8_0 + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_0 (8-bit with per-block scale)
// Y Type: F16 (half precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q8_0_f16_s1                     (single batch decode)
//     q8_0_f16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q8_0_f16_s64, q8_0_f16_s64_tc   (64-batch, smem weight tiles)
//     q8_0_f16_s128, q8_0_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_0.cuh"

// Kernel instantiation with F16 Y vectors
// K-TILE-MAJOR: Uses QK8_0_KTILE=256, QI8_0_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q8_0_f16,
    QK8_0_KTILE, QI8_0_KTILE, block_c_q8_0, VDR_Q8_0_KTILE,
    __half, __half
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<false, half, half, half, block_c_q8_0>(
//     const half*, const int4*, half*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q8_0_f16(const half* a, const int4* w, half* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<false, half, half, half, block_c_q8_0>(a, w, o, s, b, n, k, st, d);
// }}