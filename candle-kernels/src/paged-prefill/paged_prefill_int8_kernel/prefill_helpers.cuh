// Int8 element helpers, the MMA and ldmatrix fragment loaders, the tile
// constants, and `PrefillCut`.
//
// Holds the per-element int8 helpers shared with the tile decode kernel, the
// `mma_int8_m16n8k32` and ldmatrix fragment loaders, and the warp and tile
// constants such as `I8_WARPS` and `I8_COLS_PER_WARP`.
// Also holds `PrefillCut`, which says which blocks a launch runs and where a
// key chunk's state carries, with `i8_carry_base` locating the carry rows.

using fused_attn::load_a_frag_m16k32_ldmatrix;
using fused_attn::load_b_frag_n8k32_ldmatrix;
using fused_attn::mma_int8_m16n8k32;

// Per-element helpers shared with the INT8 tile decode kernel: the Q side's
// quantisation, and the pre-staged columns' cp.async fences.
using int8_elem::qt_to_f32;
using int8_elem::qt_from_f32;
using int8_elem::i8_quant;
using int8_elem::i8_cp_async16;
using int8_elem::i8_cp_commit;
using int8_elem::i8_cp_wait0;

constexpr int I8_WARPS = 8;
constexpr int I8_THREADS = I8_WARPS * 32;
constexpr int I8_TILE_TOK = 32;              // packed columns per tile
constexpr int I8_COLS_PER_WARP = I8_TILE_TOK / I8_WARPS; // columns a warp stages

// The selection ratio a launch may carry: a tile packs whole selection
// blocks, so the ratio must divide the tile. The launcher refuses the rest.
__host__ __device__ constexpr bool i8_ratio_supported(int ratio) {
    return ratio == 4 || ratio == 8 || ratio == 16 || ratio == 32;
}

// Head-dim-split warp grouping: warp = (row-tile, dim-part). A GROUP of
// `i8_dim_split` warps serves one m16 row-tile — all of them duplicate the
// (cheap) QK + softmax for those 16 rows, and each accumulates only its
// own 1/split of the output dims. That divides the o_acc register hog by
// the split, which is what makes the register budget of the target
// occupancy reachable. Staging is unchanged — K and V^T are staged once
// per block and shared through smem, so no global traffic is duplicated.
// The known cost: compute-bound shapes (long q, short prefix) pay for the
// duplicated QK (§13.3 rounds 6–9).
//
// The split is chosen so a warp always owns at most 64 output dims, i.e.
// `o_acc` never exceeds 32 FP32 registers:
//   HEAD_DIM  64/128 → split 2 → 4 row-tiles, 64 M-rows, o_acc 16/32
//   HEAD_DIM     256 → split 4 → 2 row-tiles, 32 M-rows, o_acc 32
// `__host__ __device__` because both the launcher (host, for the grid) and
// the kernel (device, for the warp partition) derive their shape from these.
__host__ __device__ constexpr int i8_dim_split(int head_dim) {
    return head_dim >= 256 ? 4 : 2;
}
__host__ __device__ constexpr int i8_row_tiles(int head_dim) {
    return I8_WARPS / i8_dim_split(head_dim);
}
__host__ __device__ constexpr int i8_m_rows(int head_dim) {
    return i8_row_tiles(head_dim) * 16;
}

// Target blocks/SM, and the per-block smem budget that follows from it at
// the 102 KB SM limit.
//
// Through HEAD_DIM 128 the staging slabs fit 25.6 KB and the kernel runs at
// the 4-blocks/SM max-occupancy point on a 64-register budget. At 256 both
// halves of that break: the tile overlay is ~40 KB (s_v8t and the raw
// staging scratch are linear in HEAD_DIM), and `q_frag` alone doubles to 32
// registers, so 64 is unreachable no matter how the output dims are split.
// 2 blocks/SM is the next residency point that fits — 42 KB of smem stays
// under the 48 KB static-shared ceiling, and the 128-register budget clears
// q_frag(32) + o_acc(32) with room for the working set.
__host__ __device__ constexpr int i8_min_blocks(int head_dim) {
    return head_dim >= 256 ? 2 : 4;
}
__host__ __device__ constexpr int i8_smem_budget(int head_dim) {
    return head_dim >= 256 ? 47 * 1024 : 25600;
}

// ============================================================================
// The cut: which blocks a launch runs, and where a key chunk's state carries
// ============================================================================

/// One launch's place in the cut (see the header). The key window itself
/// travels in `PrefillKvStage` (`chunk_lo`, `chunk_hi`), which the
/// pre-staging pass and the attention kernel share.
///
/// `carry` holds one record `[o; HEAD_DIM] ++ [m, l]` (FP32) per (query token,
/// head) of every staged sequence deeper than `chunk_positions`, up to one row
/// group of tokens each; `resume` one `{bound, tile_ord}` per block of the
/// same sequences' row group, per KV head. Both are packed in batch order over
/// those sequences only, and written in full before they are read: a block
/// reads them only in the launch after the one that left them.
struct PrefillCut {
    int blk_lo;          // first grid-x block of this launch's row group
    int group_blocks;    // grid-x blocks per row group
    int chunk_positions; // key positions per chunk: a deeper staged sequence carries
    int last_chunk;      // nonzero on the row group's last key window
    float* carry;
    uint2* resume;
};

/// Where a carrying sequence's carry rows and resume entries start: the
/// carrying sequences before `b`'s row-group tokens and blocks. Called only by
/// blocks that carry, once on the way in and once on the way out.
__device__ __forceinline__ void i8_carry_base(
    const uint32_t* __restrict__ q_lens, const uint32_t* __restrict__ kv_lens,
    int b, int min_q_len, const PrefillCut& cut, int block_m_tok,
    int64_t& tok_base, int& blk_base)
{
    const int group_tok = cut.group_blocks * block_m_tok;
    tok_base = 0;
    blk_base = 0;
    for (int i = 0; i < b; ++i) {
        const int qi = (int)q_lens[i];
        if (qi >= min_q_len && (int)kv_lens[i] > cut.chunk_positions) {
            tok_base += min(qi, group_tok);
            blk_base += min((qi + block_m_tok - 1) / block_m_tok, cut.group_blocks);
        }
    }
}
