#pragma once
// =============================================================================
// mma_wrappers.cuh — INT8 MMA inline-asm wrappers + fragment loaders.
//
// Per-arch MMA instructions and per-thread fragment sizes (PTX ISA 8.x):
//   sm_86  : mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32
//             A: 16x16 = 256 INT8 / 32 threads = 8 INT8/thread = 2 b32 regs
//             B:  8x16 = 128 INT8 / 32 threads = 4 INT8/thread = 1 b32 reg
//             C/D: 16x8 = 128 INT32 / 32 = 4 regs
//   sm_89  : mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//             A: 16x32 = 512 INT8 / 32 = 16 INT8/thread = 4 b32 regs
//             B:  8x32 = 256 INT8 / 32 =  8 INT8/thread = 2 b32 regs
//             C/D: 4 INT32 regs
//   sm_120 : same shape as sm_89.
// =============================================================================

#include "arch_traits.cuh"
#include <cstdint>

namespace fused_attn {

template<int SM_VERSION>
struct MmaTraits;

template<> struct MmaTraits<89> {
    static constexpr int A_B32 = 4;   // A regs per thread
    static constexpr int B_B32 = 2;   // B regs per thread
    static constexpr int C_REGS = 4;  // C/D regs per thread
};
template<> struct MmaTraits<120> {
    static constexpr int A_B32 = 4;
    static constexpr int B_B32 = 2;
    static constexpr int C_REGS = 4;
};
template<> struct MmaTraits<86> {
    static constexpr int A_B32 = 2;
    static constexpr int B_B32 = 1;
    static constexpr int C_REGS = 4;
};

// sm_89 / sm_120 m16n8k32 INT8.
__device__ __forceinline__ void mma_int8_m16n8k32(
    int32_t       (&d)[4],
    const uint32_t(&a)[4],
    const uint32_t(&b)[2],
    const int32_t (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// sm_86 m16n8k16 INT8.
__device__ __forceinline__ void mma_int8_m16n8k16(
    int32_t       (&d)[4],
    const uint32_t(&a)[2],
    const uint32_t(&b)[1],
    const int32_t (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%7, %8, %9, %10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// =============================================================================
// MMA fragment loaders.
//
// For sm_89 m16n8k32 INT8 with A 16x32 k-major in smem:
//   per-thread holds 16 INT8 = 4 b32 regs.
// For B 8x32 k-major in smem:
//   per-thread holds 8 INT8 = 2 b32 regs.
//
// Implementations below use direct strided loads (not ldmatrix) for
// simplicity and to keep the lane mapping inspectable. ldmatrix-based
// optimised paths can be substituted in a Phase B optimisation pass.
// =============================================================================

// Load a 16x32 INT8 A fragment (k-major). Lane t holds the row m and the row
// m+8 worth of A's data, packed as 4 .b32 (16 INT8) per the m16n8k32 layout:
//   a[0] = bytes at row(lane>>2),       cols (lane%4)*4 .. (lane%4)*4+3
//   a[1] = bytes at row(lane>>2)+8,     cols (lane%4)*4 .. (lane%4)*4+3
//   a[2] = bytes at row(lane>>2),       cols (lane%4)*4+16 .. +19
//   a[3] = bytes at row(lane>>2)+8,     cols (lane%4)*4+16 .. +19
__device__ __forceinline__ void load_a_frag_m16k32(
    uint32_t      (&a)[4],
    const int8_t* smem_a,
    int           lda_bytes,
    int           lane
) {
    int row_base = (lane >> 2);
    int col_base = (lane & 3) * 4;
    a[0] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base    ) * lda_bytes + col_base);
    a[1] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base + 8) * lda_bytes + col_base);
    a[2] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base    ) * lda_bytes + col_base + 16);
    a[3] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base + 8) * lda_bytes + col_base + 16);
}

// ldmatrix variant of the 16x32 INT8 A load (sm_75+). The m16n8k32 s8 A operand is
// 16 rows × 32 int8 = 16 × 16 b16 = exactly ldmatrix.x4's four 8×8 b16 tiles:
//   tile0 = rows 0-7  / K 0-15  -> a[0]   tile2 = rows 0-7  / K 16-31 -> a[2]
//   tile1 = rows 8-15 / K 0-15  -> a[1]   tile3 = rows 8-15 / K 16-31 -> a[3]
// Within a tile ldmatrix hands thread t the two contiguous b16 (= 4 contiguous int8)
// at (row t>>2, K (t&3)*4..+3) — byte-identical to the strided loader's a[i], no
// permute. smem_a must be PADDED (stride not a multiple of 128B) and 16B-aligned so
// the 8 tile rows land in distinct banks (else the conflict just moves to ldmatrix).
__device__ __forceinline__ void load_a_frag_m16k32_ldmatrix(
    uint32_t      (&a)[4],
    const int8_t* smem_a,
    int           lda_bytes,
    int           lane
) {
    const int tile_idx    = lane >> 3;           // 0..3 (lanes 0-7,8-15,16-23,24-31)
    const int row_in_tile = lane & 7;            // 0..7
    const int m_offset = (tile_idx & 1) * 8;     // rows 0-7 or 8-15
    const int k_offset = (tile_idx >> 1) * 16;   // int8 K 0-15 or 16-31
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(
        smem_a + (int64_t)(m_offset + row_in_tile) * lda_bytes + k_offset));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr)
    );
}

// Load an 8x32 INT8 B fragment. Per PTX ISA m16n8k32 the B operand uses the
// SAME (groupID, threadID) lane decomposition as A — groupID = laneid>>2 selects
// the column n (0..7), threadID = laneid&3 selects the k-quad — NOT (lane%8,
// lane/8). Getting this wrong makes the tensor core contract mismatched
// elements, silently corrupting every QK dot.
//   b[0] = bytes at row(n = lane>>2), cols (lane&3)*4 .. +3
//   b[1] = bytes at row(n = lane>>2), cols (lane&3)*4 + 16 .. +19
__device__ __forceinline__ void load_b_frag_n8k32(
    uint32_t      (&b)[2],
    const int8_t* smem_b,
    int           ldb_bytes,
    int           lane
) {
    int row = lane >> 2;
    int col_base = (lane & 3) * 4;
    b[0] = *reinterpret_cast<const uint32_t*>(smem_b + (int64_t)row * ldb_bytes + col_base);
    b[1] = *reinterpret_cast<const uint32_t*>(smem_b + (int64_t)row * ldb_bytes + col_base + 16);
}

// sm_86 (m16n8k16) variants — A is 16x16 = 2 b32, B is 8x16 = 1 b32.
__device__ __forceinline__ void load_a_frag_m16k16(
    uint32_t      (&a)[2],
    const int8_t* smem_a,
    int           lda_bytes,
    int           lane
) {
    int row_base = (lane >> 2);
    int col_base = (lane & 3) * 4;
    a[0] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base    ) * lda_bytes + col_base);
    a[1] = *reinterpret_cast<const uint32_t*>(smem_a + (int64_t)(row_base + 8) * lda_bytes + col_base);
}

__device__ __forceinline__ void load_b_frag_n8k16(
    uint32_t      (&b)[1],
    const int8_t* smem_b,
    int           ldb_bytes,
    int           lane
) {
    // Same B-operand decomposition as m16n8k32 (see load_b_frag_n8k32):
    // n = lane>>2, k = (lane&3)*4.
    int row = lane >> 2;
    int col_base = (lane & 3) * 4;
    b[0] = *reinterpret_cast<const uint32_t*>(smem_b + (int64_t)row * ldb_bytes + col_base);
}

} // namespace fused_attn
