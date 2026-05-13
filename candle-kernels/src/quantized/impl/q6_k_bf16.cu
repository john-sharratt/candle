// =============================================================================
// Q6_K + BF16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q6_K (6-bit with per-block scales, K-quant)
// Y Type: BF16 (bfloat16 activations)
// Block type: block_c_q6_K (typedef to block_c_q6_K_k128, 128 bytes)
// Elements per block: 128 (16 threads × 8 elements)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q6_k_bf16_s1                      (single batch decode)
//     q6_k_bf16_s8                      (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q6_k_bf16_s64, q6_k_bf16_s64_tc   (64-batch, smem weight tiles)
//     q6_k_bf16_s128, q6_k_bf16_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_BF16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q6_k.cuh"

// K/128 format: 128 elements per block, 32 ints of data
constexpr int QK6_K_K128 = 128;
constexpr int QI6_K_K128 = 32;
// VDR=2 gives 16 threads per quant block (128 / (4 * 2) = 16)
constexpr int VDR_Q6_K_K128 = 2;

// Kernel instantiation with BF16 Y vectors
// K/128 format requires vdr=2 (16 threads per K/128 block)
INSTANTIATE_KERNELS(
    q6_k_bf16,
    QK6_K_K128, QI6_K_K128, block_c_q6_K, VDR_Q6_K_K128,
    __nv_bfloat16, __nv_bfloat16
)

// =============================================================================
// MARLIN TENSOR CORE KERNEL - Q6_K + BF16
// =============================================================================
// DISABLED: dequant_k64 not yet implemented for this quant type
// Uses block_c_q6_K (K/64 embedded-scale layout) for Marlin tensor core GEMM.
// BF16 activations and output.
// =============================================================================

// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"

// // Explicit template instantiation for Q6_K with BF16
// template int marlin::marlin_launch_simple<true, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, block_c_q6_K>(
//     const __nv_bfloat16*, const int4*, __nv_bfloat16*, const void*,
//     int, int, int, cudaStream_t, int);

// namespace marlin {

// int marlin_q6_K_bf16(
//     const __nv_bfloat16* activations, const int4* weights, __nv_bfloat16* output, const void* scales,
//     int batch_size, int out_features, int in_features, cudaStream_t stream, int dev)
// {
//     return marlin_launch_simple<true, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, block_c_q6_K>(
//         activations, weights, output,
//         scales, batch_size, out_features, in_features, stream, dev);
// }

// }  // namespace marlin
