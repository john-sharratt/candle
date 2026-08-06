// =============================================================================
// MXFP4_KO + F32 KERNEL INSTANTIATION  (q8a128 INT8-MMA path)
// =============================================================================
// Native-MXFP4 exponent-collapse weights × q8a128 int8 activations on the m16n8k32
// tensor core, F32 output. MXFP4_KO is INT8-ONLY (no FP GEMX kernel): the routed
// DeepSeek-V4 experts are trained MXFP4 and only ever run through this path. The two
// int8 kernels (grouped + dense) are fully generic over the block type — the only
// format-specific code is the collapse dequant in ../loader/mxfp4.cuh.
//
// K/128 format parameters mirror Q4_KO (4-bit, 128-elem K tiles): qk=128, qi=32, vdr=2.
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/mxfp4.cuh"

constexpr int QK_MXFP4_K128 = 128;
constexpr int QI_MXFP4_K128 = 32;
constexpr int VDR_MXFP4_K128 = 2;

// Grouped (MoE) + dense (mode-1, Bm=16) int8 entries: emits
// mxfp4_ko_int8_f32_grouped and mxfp4_ko_int8_f32_dense.
INSTANTIATE_KERNEL_GROUPED_INT8(
    mxfp4_ko_int8_f32,
    QK_MXFP4_K128, QI_MXFP4_K128, block_c_mxfp4, VDR_MXFP4_K128,
    float
)

// Mode-2 dense variant (N_SUB=2, Bm=32): the weight chunk's dequant is reused across 2
// token sub-tiles per block → half the weight re-reads, for large-M (prefill) scaling.
// Host launches with grid.x = ceil(total_batch / 32) (vs /16 for mode-1).
extern "C" __global__ void LAUNCH_BOUNDS_TC16 mxfp4_ko_int8_f32_dense_m2(
    const void* __restrict__ weights,
    const block_q8a128* __restrict__ vy, float* __restrict__ dst,
    const int ncols_x, const int nrows_x, const int total_batch,
    const int y_stride, const int dst_stride) {
    grouped_tc::quantized_matmul_dense_entry_int8<QK_MXFP4_K128, QI_MXFP4_K128, block_c_mxfp4,
                                                  VDR_MXFP4_K128, float, 2>(
        reinterpret_cast<const block_compact_t<block_c_mxfp4>*>(weights),
        vy, dst, ncols_x, nrows_x, total_batch, y_stride, dst_stride);
}
