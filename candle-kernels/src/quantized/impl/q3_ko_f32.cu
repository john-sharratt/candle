// =============================================================================
// Q3_KO KERNEL INSTANTIATION  (q8a128 INT8-MMA path)
// =============================================================================
// 3-bit affine KO weights × q8a128 int8 activations on the m16n8k32 tensor core.
// Q3_KO is INT8-ONLY (no FP GEMX kernel), like the other KO twins: `Q3_K`'s
// same-width twin, read through the maintained per-128 (scale, min) int8 fold.
// The int8 kernels (grouped + dense) are generic over the block type — the only
// format-specific code is the crumb+hi unpack in ../loader/q3_KO.cuh.
//
// The grouped (MoE) entry stores F32 (feeds the SwiGLU requantisation); the
// dense entries store at the consumer's activation dtype (f16/bf16/f32).
//
// K/128 format parameters mirror Q4_KO (128-elem K tiles): qk=128, qi=32, vdr=2.
// =============================================================================

#define Y_TYPE_F32

#include "kernel_instantiate.cuh"
#include "common.cuh"
#include "../loader/q3_KO.cuh"

constexpr int QK_Q3_KO_K128 = 128;
constexpr int QI_Q3_KO_K128 = 32;
constexpr int VDR_Q3_KO_K128 = 2;

INSTANTIATE_KERNEL_GROUPED_INT8(
    q3_ko_int8_f32,
    QK_Q3_KO_K128, QI_Q3_KO_K128, block_c_q3_KO, VDR_Q3_KO_K128,
    float
)
INSTANTIATE_KERNEL_GROUPED_INT8_M4(
    q3_ko_int8_f32,
    QK_Q3_KO_K128, QI_Q3_KO_K128, block_c_q3_KO, VDR_Q3_KO_K128,
    float
)
INSTANTIATE_KERNEL_GROUPED_INT8_M8(
    q3_ko_int8_f32,
    QK_Q3_KO_K128, QI_Q3_KO_K128, block_c_q3_KO, VDR_Q3_KO_K128,
    float
)
INSTANTIATE_KERNEL_DENSE_INT8_ALL(
    q3_ko_int8,
    QK_Q3_KO_K128, QI_Q3_KO_K128, block_c_q3_KO, VDR_Q3_KO_K128
)
// Mode-2 dense (N_SUB=2, Bm=32): the weight chunk's dequant is reused across 2
// token sub-tiles per block → half the weight re-reads, for large-M (prefill)
// scaling. Host launches with grid.x = ceil(total_batch / 32) (vs /16 for mode-1).
INSTANTIATE_KERNEL_DENSE_INT8_M2_ALL(
    q3_ko_int8,
    QK_Q3_KO_K128, QI_Q3_KO_K128, block_c_q3_KO, VDR_Q3_KO_K128
)
