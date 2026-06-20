// =============================================================================
// Q6_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q6_K (6-bit K-quant with per-block scales)
// Y Type: F32 (float32 activations)
// Block type: block_c_q6_K (typedef to block_c_q6_K_k128, 128 bytes)
// Elements per block: 128 (16 threads × 8 elements)
//
// Purpose: High-precision reference path for validation
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q6_K.cuh"

// K/128 format: 128 elements per block, 32 ints of data
constexpr int QK6_K_K128 = 128;
constexpr int QI6_K_K128 = 32;
// VDR=2 gives 16 threads per quant block (128 / (4 * 2) = 16)
constexpr int VDR_Q6_K_K128 = 2;

// Kernel instantiation with F32 Y vectors and F32 output
// K/128 format requires vdr=2 (16 threads per K/128 block)
INSTANTIATE_KERNELS(
    q6_k_f32,
    QK6_K_K128, QI6_K_K128, block_c_q6_K, VDR_Q6_K_K128,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q6_K (6-bit symmetric weights, per-16
// scales re-binned to per-32, centered int8, F32 output). See grouped_tc_int8.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q6_k_int8_f32,
    QK6_K_K128, QI6_K_K128, block_c_q6_K, VDR_Q6_K_K128,
    float
)

// Q6_KO TC path — same INT8-MMA kernels reading the Q6_KO block (byte-identical to
// Q6_K; the compact layout is already in ordered form).
INSTANTIATE_KERNEL_GROUPED_INT8(
    q6_ko_int8_f32,
    QK6_K_K128, QI6_K_K128, block_c_q6_KO, VDR_Q6_K_K128,
    float
)

// Mode-2 dense variant (N_SUB=2, Bm=32): weight-reuse loop for large-M (prefill); selected when the
// host passes force_mode2=1 (grid.x = ceil(total_batch/32)).
extern "C" __global__ void LAUNCH_BOUNDS_TC16 q6_ko_int8_f32_dense_m2(
    const void* __restrict__ weights,
    const block_q8a128* __restrict__ vy, float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int total_batch,
    const int y_stride, const int dst_stride) {
    grouped_tc::quantized_matmul_dense_entry_int8<QK6_K_K128, QI6_K_K128, block_c_q6_KO,
                                                  VDR_Q6_K_K128, float, 2>(
        reinterpret_cast<const block_compact_t<block_c_q6_KO>*>(weights),
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride);
}

// =============================================================================
// MARLIN TENSOR CORE KERNEL - Q6_K + F32
// =============================================================================
// DISABLED: dequant_k64 not yet implemented for this quant type
// Uses block_c_q6_K (K/64 embedded-scale layout) for Marlin tensor core GEMM.
// F32 output for high-precision validation.
// =============================================================================

// #include "../marlin.cuh"
// #include "../marlin_dispatch.cuh"

// // Explicit template instantiation for Q6_K with F32 output
// template int marlin::marlin_launch_simple<true, half, float, float, block_c_q6_K>(
//     const float*, const int4*, float*, const void*,
//     int, int, int, cudaStream_t, int);

// namespace marlin {

// int marlin_q6_K_f32(
//     const float* activations, const int4* weights, float* output, const void* scales,
//     int batch_size, int out_features, int in_features, cudaStream_t stream, int dev)
// {
//     return marlin_launch_simple<true, half, float, float, block_c_q6_K>(
//         activations, weights, output,
//         scales, batch_size, out_features, in_features, stream, dev);
// }

// }  // namespace marlin
