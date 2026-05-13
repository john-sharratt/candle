// =============================================================================
// Q4_K + F16 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q4_K (4-bit with per-block scales, K-quant) in K/128 format
// Y Type: F16 (half precision activations)
//
// K/128 format: 128 elements per block, 16 threads per block, 8 elements per thread
// Block type: block_c_q4_K (typedef to block_c_q4_K_k128, 80 bytes)
//
// Generates batch-specialized kernels:
//   Register-only (memory-bound):
//     q4_k_f16_s1  (single batch decode)
//     q4_k_f16_s8  (octet batch, 8× weight reuse)
// =============================================================================

#define Y_TYPE_F16

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q4_k.cuh"

// K/128 format parameters:
// qk = 128 elements per block
// qi = 32 (16 threads × vdr=2)
// vdr = 2 (vector dimension register)
constexpr int QK4_K_K128 = 128;
constexpr int QI4_K_K128 = 32;
constexpr int VDR_Q4_K_K128 = 2;

// Kernel instantiation with F16 Y vectors
INSTANTIATE_KERNELS(
    q4_k_f16,
    QK4_K_K128, QI4_K_K128, block_c_q4_K, VDR_Q4_K_K128,
    __half, __half
)

// =============================================================================
// MARLIN TENSOR CORE KERNEL - Q4_K + F16
// =============================================================================
// DISABLED: dequant_k64 not yet implemented for this quant type
// Uses block_c_q4_K (8-byte K-tile-major layout) for Marlin tensor core GEMM.
// This is separate from the MMVQ kernels above which use the original block_q4_K.
// Note: block_c_q4_K is typedef'd to block_c_q4_K_ktile (8B K-tile struct)
// =============================================================================

// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"

// // Explicit template instantiation for Q4_K K-tile-major layout
// template int marlin::marlin_launch_simple<true, half, half, half, block_c_q4_K>(
//     const half*, const int4*, half*, const void*,
//     int, int, int, cudaStream_t, int);

// namespace marlin {

// int marlin_q4_K_f16(
//     const half* activations, const int4* weights, half* output, const void* scales,
//     int batch_size, int out_features, int in_features, cudaStream_t stream, int dev)
// {
//     return marlin_launch_simple<true, half, half, half, block_c_q4_K>(
//         activations, weights, output,
//         scales, batch_size, out_features, in_features, stream, dev);
// }

// }  // namespace marlin
