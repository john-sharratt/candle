// =============================================================================
// Q8_K + F32 KERNEL INSTANTIATION
// =============================================================================
// Quantization: Q8_K (8-bit K-quant with 256-element super-blocks)
// Y Type: F32 (single precision activations)
//
// Generates 8 kernels:
//   Register-only (memory-bound):
//     q8_k_f32_s1                     (single batch decode)
//     q8_k_f32_s8                     (octet batch, 8× weight reuse)
//   Shared memory (transitional):
//     q8_k_f32_s64, q8_k_f32_s64_tc   (64-batch, smem weight tiles)
//     q8_k_f32_s128, q8_k_f32_s128_tc (128-batch, high-smem HW)
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q8_K.cuh"

// Kernel instantiation with F32 Y vectors
// K/128 format: Uses QK8_K_KTILE=128, QI8_K_KTILE=16, VDR=1 → 16 threads/K/128 block
INSTANTIATE_KERNELS(
    q8_k_f32,
    QK8_K_KTILE, QI8_K_KTILE, block_c_q8_K, VDR_Q8_K_KTILE,
    float, float
)

// q8a128 TC path — INT8-MMA grouped matmul for Q8_K (8-bit symmetric weights, single
// block scale, F32 output). See grouped_tc_int8.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q8_k_int8_f32,
    QK8_K_KTILE, QI8_K_KTILE, block_c_q8_K, VDR_Q8_K_KTILE,
    float
)

// Q8_KO TC path — same INT8-MMA kernels reading the byte-permuted Q8_KO block.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q8_ko_int8_f32,
    QK8_K_KTILE, QI8_K_KTILE, block_c_q8_KO, VDR_Q8_K_KTILE,
    float
)

// Mode-2 dense variant (N_SUB=2, Bm=32): weight-reuse loop for large-M (prefill); selected by ytype==4.
extern "C" __global__ void LAUNCH_BOUNDS_TC16 q8_ko_int8_f32_dense_m2(
    const void* __restrict__ weights,
    const block_q8a128* __restrict__ vy, float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int total_batch,
    const int y_stride, const int dst_stride) {
    grouped_tc::quantized_matmul_dense_entry_int8<QK8_K_KTILE, QI8_K_KTILE, block_c_q8_KO,
                                                  VDR_Q8_K_KTILE, float, 2>(
        reinterpret_cast<const block_compact_t<block_c_q8_KO>*>(weights),
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride);
}
