// =============================================================================
// Q5_0 + BF16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q5_0 (5-bit with per-block scale)
// Y Type: BF16 (bfloat16 activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q5_0_bf16_s1                      (single batch decode)
//     q5_0_bf16_s8                      (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q5_0_bf16_s64, q5_0_bf16_s64_tc   (64-batch, smem weight tiles)
//     q5_0_bf16_s128, q5_0_bf16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_BF16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q5_0.cuh"

// Kernel instantiation with BF16 Y vectors
// K-TILE-MAJOR: Uses QK5_0_KTILE=256, QI5_0_KTILE=32, VDR=2 → 16 threads/super-block
INSTANTIATE_KERNELS(
    q5_0_bf16,
    QK5_0_KTILE, QI5_0_KTILE, block_c_q5_0, VDR_Q5_0_KTILE,
    __nv_bfloat16, __nv_bfloat16
)

// DISABLED: dequant_k64 not yet implemented for this quant type
// MARLIN TENSOR CORE KERNEL
// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"
// template int marlin::marlin_launch_simple<false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, block_c_q5_0>(
//     const __nv_bfloat16*, const int4*, __nv_bfloat16*, const void*, int, int, int, cudaStream_t, int);
// namespace marlin {
// int marlin_q5_0_bf16(const __nv_bfloat16* a, const int4* w, __nv_bfloat16* o, const void* s,
//     int b, int n, int k, cudaStream_t st, int d) {
//     return marlin_launch_simple<false, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, block_c_q5_0>(a, w, o, s, b, n, k, st, d);
// }}