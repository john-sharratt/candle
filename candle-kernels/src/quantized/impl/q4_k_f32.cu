// =============================================================================
// Q4_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q4_K (4-bit with per-block scales, K-quant) in K/128 format
// Y Type: F32 (float32 activations)
//
// Purpose: High-precision reference path for validation
// Uses F32 activations and F32 output for comparing against baseline dequant+matmul.
// This allows tighter tolerance tests to verify kernel correctness independently
// of precision loss from FP16/BF16 conversion.
//
// K/128 format: 128 elements per block, 16 threads per block, 8 elements per thread
// Block type: block_c_q4_K (typedef to block_c_q4_K_k128, 80 bytes)
// =============================================================================

#define Y_TYPE_F32

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

// Kernel instantiation with F32 Y vectors and F32 output
INSTANTIATE_KERNELS(
    q4_k_f32,
    QK4_K_K128, QI4_K_K128, block_c_q4_K, VDR_Q4_K_K128,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul: q8a128 activations (raw int8 qs) ×
// Q4_K weights (raw 4-bit nibbles) on the m16n8k32 int8 tensor core, deferred-scale
// fold, F32 output. See grouped_tc_int8 in kernel.cuh.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q4_k_int8_f32,
    QK4_K_K128, QI4_K_K128, block_c_q4_K, VDR_Q4_K_K128,
    float
)

// Q4_KO TC path — same INT8-MMA grouped/dense kernels, reading the byte-permuted
// Q4_KO block (qs contiguous, scales at the tail). Reuses every bit of the Q4_K
// int8 path via the inheriting gemx_dequant_traits<block_c_q4_KO> trait.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q4_ko_int8_f32,
    QK4_K_K128, QI4_K_K128, block_c_q4_KO, VDR_Q4_K_K128,
    float
)

// Mode-2 dense variant (N_SUB=2, Bm=32): the weight chunk's dequant is reused across 2 token
// sub-tiles per block → half the weight re-reads. For large-M (prefill) scaling experiments.
// Host launches with grid.x = ceil(total_batch / 32) (vs /16 for mode-1).
extern "C" __global__ void LAUNCH_BOUNDS_TC16 q4_ko_int8_f32_dense_m2(
    const void* __restrict__ weights,
    const block_q8a128* __restrict__ vy, float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int total_batch,
    const int y_stride, const int dst_stride) {
    grouped_tc::quantized_matmul_dense_entry_int8<QK4_K_K128, QI4_K_K128, block_c_q4_KO,
                                                  VDR_Q4_K_K128, float, 2>(
        reinterpret_cast<const block_compact_t<block_c_q4_KO>*>(weights),
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride);
}

// =============================================================================
// GEMX TENSOR CORE KERNEL - Q4_K + F32
// =============================================================================
// DISABLED: dequant_k64 not yet implemented for this quant type
// Uses block_c_q4_K (8-byte K-tile-major layout) for GEMX tensor core GEMM.
// F32 output for high-precision validation.
// =============================================================================

// #include "../gemx.cuh"
// #include "../gemx_dispatch.cuh"

// // Explicit template instantiation for Q4_K with F32 output
// template int gemx::gemx_launch_simple<true, half, float, float, block_c_q4_K>(
//     const float*, const int4*, float*, const void*,
//     int, int, int, cudaStream_t, int);

// namespace gemx {

// int gemx_q4_K_f32(
//     const float* activations, const int4* weights, float* output, const void* scales,
//     int batch_size, int out_features, int in_features, cudaStream_t stream, int dev)
// {
//     return gemx_launch_simple<true, half, float, float, block_c_q4_K>(
//         activations, weights, output,
//         scales, batch_size, out_features, in_features, stream, dev);
// }

// }  // namespace gemx
