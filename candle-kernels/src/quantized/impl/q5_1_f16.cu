// =============================================================================
// Q5_1 + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_1 (5-bit with per-block scale and min)
// Y Type: F16 (half precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q5_1_f16_s1                     (single batch decode)
//     q5_1_f16_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q5_1_f16_s64, q5_1_f16_s64_tc   (64-batch, smem weight tiles)
//     q5_1_f16_s128, q5_1_f16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_1.cuh"

// Kernel instantiation with F16 Y vectors
// K-TILE-MAJOR: Uses QK5_1_KTILE=256, QI5_1_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q5_1_f16,
    QK5_1_KTILE, QI5_1_KTILE, block_c_q5_1, VDR_Q5_1_KTILE,
    __half, __half
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// GEMX TENSOR CORE KERNEL
// #include "../gemx.cuh"
// #include "../gemx_dispatch.cuh"
// template int gemx::gemx_launch_simple<false, half, half, half, block_c_q5_1>(
//     const half*, const int4*, half*, const void*, int, int, int, cudaStream_t, int);
// namespace gemx {
// int gemx_q5_1_f16(const half* a, const int4* w, half* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return gemx_launch_simple<false, half, half, half, block_c_q5_1>(a, w, o, s, b, n, k, st, d);
// }}