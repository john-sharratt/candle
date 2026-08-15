// =============================================================================
// Q2_KO + F32 KERNEL INSTANTIATION  (q8a128 INT8-MMA path)
// =============================================================================
// 2-bit affine KO weights × q8a128 int8 activations on the m16n8k32 tensor core, F32
// output. Q2_KO is INT8-ONLY (no FP GEMX kernel), like the other KO twins: the smallest
// KO weight, read through the maintained per-128 (scale, min) int8 fold. The two int8
// kernels (grouped + dense) are generic over the block type — the only format-specific
// code is the 2-bit crumb unpack in ../loader/q2_KO.cuh.
//
// K/128 format parameters mirror Q4_KO (128-elem K tiles): qk=128, qi=32, vdr=2.
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q2_KO.cuh"

constexpr int QK_Q2_KO_K128 = 128;
constexpr int QI_Q2_KO_K128 = 32;
constexpr int VDR_Q2_KO_K128 = 2;

// Grouped (MoE) + dense (mode-1, Bm=16) int8 entries: emits
// q2_ko_int8_f32_grouped and q2_ko_int8_f32_dense.
INSTANTIATE_KERNEL_GROUPED_INT8(
    q2_ko_int8_f32,
    QK_Q2_KO_K128, QI_Q2_KO_K128, block_c_q2_KO, VDR_Q2_KO_K128,
    float
)

// Mode-2 dense variant (N_SUB=2, Bm=32): the weight chunk's dequant is reused across 2
// token sub-tiles per block → half the weight re-reads, for large-M (prefill) scaling.
// Host launches with grid.x = ceil(total_batch / 32) (vs /16 for mode-1).
extern "C" __global__ void LAUNCH_BOUNDS_TC16 q2_ko_int8_f32_dense_m2(
    const void* __restrict__ weights,
    const block_q8a128* __restrict__ vy, float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int total_batch,
    const int y_stride, const int dst_stride) {
    grouped_tc::quantized_matmul_dense_entry_int8<QK_Q2_KO_K128, QI_Q2_KO_K128, block_c_q2_KO,
                                                  VDR_Q2_KO_K128, float, 2>(
        reinterpret_cast<const block_compact_t<block_c_q2_KO>*>(weights),
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride);
}
