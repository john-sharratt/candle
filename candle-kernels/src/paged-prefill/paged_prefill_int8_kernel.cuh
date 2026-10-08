/*
 * ============================================================================
 * INT8 PREFIX-ATTENTION PREFILL KERNEL
 * ============================================================================
 *
 * The `docs/archived/prefill_optimization.md` kernel: causal prefill attention over a
 * palette-quantized paged KV prefix, computed with INT8 m16n8k32 tensor-core
 * MMA for both Q·Kᵀ and P·V — the compressed domain is the compute domain.
 *
 * Structure (vs the FP16 `paged_prefill_attn_fwd_chunks_kernel`):
 *
 *  - GQA-PACKED M: an MMA M-row is a (query-token, head-in-group) pair.
 *    One block serves ALL query heads of one KV head — the K/V tile is
 *    loaded once per group instead of once per head-block. The 8 warps
 *    partition into (row-tile, dim-part): M_ROWS = 64 at HEAD_DIM ≤ 128
 *    (4 m16 row-tiles, each served by a warp PAIR) and 32 at HEAD_DIM 256
 *    (2 row-tiles, each served by a QUARTET), BLOCK_M_TOK = M_ROWS / hpg
 *    tokens.
 *
 *  - PACKED TILES OVER A BLOCK WALK: a KV tile is 32 SELECTED positions in
 *    ascending order, not 32 consecutive ones. The block's query rows
 *    between them select a bounded set of `ratio`-position selection
 *    blocks whatever the depth (`qsa_walk.cuh` merges the rows' entry
 *    lists); each tile packs the next 32/ratio of them, so the tile count
 *    is bounded by the rows' combined budget rather than the prefix
 *    length. A launch without a selection walks 32-position blocks — the
 *    dense causal read, in the same loop. Per-row masking inside a tile is
 *    a bit test: the walk reports which cells of each block each row
 *    selects, and the compute phase ANDs that with the causal horizon.
 *    Under split-KV a launch of sparse rows hands each split a contiguous
 *    range of positions to seek to and walk; one with a dense row deals
 *    tiles round-robin (see the ownership note before the tile loop).
 *
 *  - PER-WARP COLUMN STAGING: warp w owns tile columns 4w..4w+3 and
 *    decodes each straight from its source — a fresh token's packed
 *    input rows, or a sealed token's arena quant blocks / dtype spans
 *    through that token's slice metadata (per-warp palette bases and
 *    rank bytes, rebound only when the column's slice changes). Columns
 *    of one tile may come from different chunks, so the tile carries no
 *    single palette table; every decode is element-wise through the
 *    arena accessors, and K is RoPEd + requantised per (token, window)
 *    while V is stashed as FP16 for a per-dim requant after the barrier
 *    (`column_stage.cuh`).
 *
 *  - PRE-STAGED COLUMNS (the `PRESTAGED` instantiation): a bulk launch
 *    serves a couple of query tokens per block, so every block that
 *    selects a position decodes it again — the same row thousands of
 *    times per layer. Its long sequences are staged once per position
 *    ahead of the kernel (`kv_prestage_kernel.cuh`), by the same decode
 *    over four-aligned quads, and a warp whose four columns are such a
 *    quad copies them out of the stage with `cp.async` instead. A warp
 *    whose columns start mid-quad (a page layout's short block moves a
 *    selected block off the grid) decodes them itself, as the staged
 *    quad would have grouped its positions differently. Either way the
 *    tile holds the same bytes, and nothing after the staging barrier
 *    knows which way they came. Few-row launches (a verify window over a
 *    long prefix) stage nothing: they touch too few positions to pay for
 *    staging every one.
 *
 *  - FRESH TOKENS FROM THE INPUTS: the q_len new tokens are staged straight
 *    from the packed q/k/v tensors (never read back from the arena); the
 *    arena write of their K/V is an independent pre-pass (z == 0 only).
 *
 * Quantization grid (independent of the arena's palette routing):
 *    Q:  int8 per (M-row, 32-dim window)   — natural dim order
 *    K:  int8 per (token, 32-dim window)   — natural dim order, post-RoPE
 *    P:  int8 per row, fixed scale 1/127   (P ∈ (0, 1] after online softmax)
 *    V:  int8 per (natural dim, tile)      — requant max-abs over the tile's
 *                                            32 packed columns
 *  QK epilogue: acc_f32 += i32(window) · qs[row][w] · ks[tok][w]
 *  PV epilogue: o_f32   += i32 · (1/127) · vs[dim]
 *  The O accumulator and V^T slab are NATURAL-dim indexed — palette rank
 *  space is per-slice and cannot host a cross-tile accumulator.
 *
 * Scope: HEAD_DIM % 64 == 0 in [64, 256] (in-thread RoPE pairing); the
 * selection ratio must divide the 32-column tile (4, 8, 16, 32).
 * ============================================================================
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../arena_table.cuh"
#include "../grow_scratch.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../convert/convert_all.cuh"
#include "../mma/mma_wrappers.cuh"
#include "../convert/int8_elem.cuh"
#include "kv_store.cuh"
#include "column_stage.cuh"
#include "kv_stage.cuh"
#include "kv_prestage_kernel.cuh"
// QSA block-sparse selection — one row per PACKED QUERY (`q_start + token`).
// Null for every model that reads the whole causal prefix.
#include "../qsa_select.cuh"
#include "qsa_walk.cuh"
#include "rope_hoist.cuh"

namespace prefill_int8 {

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
// The kernel
// ============================================================================

// minBlocks pins the register budget: 4 blocks × 256 threads at 64
// regs/thread through HEAD_DIM 128 (the max-occupancy configuration, 67%
// theoretical; 6 blocks would need ≤42 regs, unreachable past o_acc's 32),
// 2 blocks × 128 regs at 256. The Q-fragment drain, the union smem arena,
// and the deliberately register-lean staging (recompute lambdas,
// smem-resident Q scales, two-pass V requant, serialized palette loops)
// are what make the budget close with only a small residual spill.
//
// `PRESTAGED` is the staging policy: false reads every column from its
// source; true takes a staged sequence's four-aligned column quads from
// `stage` (see the header).
template <typename QT, int HEAD_DIM, bool PRESTAGED>
__global__ void __launch_bounds__(I8_THREADS, i8_min_blocks(HEAD_DIM))
paged_prefill_int8_kernel(
    const QT* __restrict__ q,          // [total_q, n_head, HD] packed, unrotated
    const QT* __restrict__ k_packed,   // [total_q, n_kv_head, HD] packed, unrotated
    const QT* __restrict__ v_packed,   // [total_q, n_kv_head, HD]
    const uint8_t* __restrict__ headers_ptr,
    const uint32_t* __restrict__ cu_seqlens_q,
    const uint32_t* __restrict__ q_lens,
    const uint32_t* __restrict__ kv_lens,
    QT* __restrict__ out,              // [total_q, n_head, HD]
    int batch_size,
    int n_head,
    int n_kv_head,
    float softmax_scale,
    const RopeRungs rungs,
    int rope_interleaved,               // 0 = half-split pairing, 1 = interleaved (LLaMA)
    // Split-KV: grid.z = batch_size × num_splits. Shard s of a sequence
    // processes tiles (sealed AND fresh — one shared ordinal space) with
    // ordinal ≡ s (mod num_splits). num_splits == 1 stores O directly;
    // otherwise each shard emits
    // an un-normalized (ΣpV, m, l) partial into `partials`
    // [total_q·n_head rows][num_splits][HEAD_DIM + 2] and the combine kernel
    // merges them (base-e log-sum-exp).
    int num_splits,
    float* __restrict__ partials,
    QsaSel sel,
    const PrefillKvStage stage
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 prefill: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;       // QK k-step windows (also dims/lane)
    constexpr int PV_SLICES = HEAD_DIM / 8;    // PV n-slices (output dims per mma)
    constexpr int SUB = HEAD_DIM / N_PALETTE;  // palette band width
    // A lane's rank bytes pack (palette, rank) as p<<6 | rank.
    static_assert(N_PALETTE == 4, "rank byte packs the palette into 2 bits");
    static_assert(SUB <= 64, "rank needs 6 bits");

    constexpr int DIM_SPLIT = i8_dim_split(HEAD_DIM);
    constexpr int I8_ROW_TILES = i8_row_tiles(HEAD_DIM);
    constexpr int I8_M_ROWS = i8_m_rows(HEAD_DIM);
    static_assert(I8_ROW_TILES * DIM_SPLIT == I8_WARPS,
                  "warps must partition exactly into (row-tile, dim-part)");

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_tile = warp / DIM_SPLIT; // which m16 row-tile this warp serves
    const int dim_part = warp % DIM_SPLIT; // which output-dim slice it accumulates
    const int batch_idx = (int)blockIdx.z / num_splits;
    const int split_idx = (int)blockIdx.z % num_splits;
    const int kv_head_idx = (int)blockIdx.y;
    if (batch_idx >= batch_size || kv_head_idx >= n_kv_head) return;

    const SlotHeader& slot_hdr = get_slot_header(headers_ptr, batch_idx);
    // The sequence's own rung: a block never spans two sequences, so nothing
    // rung-dependent here is shared with another sequence's rows.
    const RopeView rope = rope_view(rungs, slot_hdr.rope_rung);

    const int q_start = (int)cu_seqlens_q[batch_idx];
    const int q_len = (int)q_lens[batch_idx];
    const int kv_len = (int)kv_lens[batch_idx];
    int prefix_len = kv_len - q_len;
    if (prefix_len < 0) prefix_len = 0;

    int hpg = n_head / n_kv_head;
    if (hpg <= 0) hpg = 1;
    if (hpg > I8_M_ROWS) return; // unsupported (production hpg = 8)
    const int block_m_tok = I8_M_ROWS / hpg; // tokens covered per block
    const int rows_used = block_m_tok * hpg; // ≤ I8_M_ROWS; rows beyond are idle
    const int t0 = (int)blockIdx.x * block_m_tok;
    if (t0 >= q_len) return;
    const int first_q_head = kv_head_idx * hpg;

    // Whether this block's sequence was pre-staged, and where its rows start
    // in each head's run of the stage. Block-uniform.
    bool staged = false;
    int64_t stage_base = 0;
    if constexpr (PRESTAGED) {
        staged = q_len >= stage.min_q_len;
        if (staged) stage_base = kv_stage_seq_base(q_lens, kv_lens, batch_idx, stage.min_q_len);
    }

    // ------------------------------------------------------------------
    // Shared memory: ONE union arena, sized for 4 blocks/SM (the 25.6 KB
    // per-block budget at the 102 KB SM limit — the max-occupancy target).
    //
    // The arena has two overlays with disjoint lifetimes:
    //   PROLOGUE overlay (block start only): s_q8 + s_q_scale. Q is staged
    //     here once, then DRAINED TO REGISTERS (the q8-matmul trick at
    //     block scope: Q is constant across the tile loop, and a warp only
    //     ever reads its own 16 rows as N_WIN ldmatrix A-fragments = 16
    //     registers + scales). After the drain barrier the whole region is
    //     dead and the tile overlay reuses its bytes.
    //   TILE overlay (per tile): the staging→compute handoff slabs
    //     (s_k8/s_v8t + scales) and the fresh∪p8 scratch. The slabs span
    //     the staging barrier so they cannot union among themselves, but
    //     all of them may alias the dead prologue.
    //
    // +16-byte row pads on the MMA slabs (the q8-matmul KI8_STRIDE
    // convention): a multiple of 16 keeps every row address ldmatrix-legal,
    // while NOT being a multiple of 128 rotates the 8 tile rows across
    // banks. Scale tables are FP16 (max-abs magnitudes; ~0.05% error is
    // noise under int8's 0.4%).
    // ------------------------------------------------------------------
    constexpr int Q8_LD = HEAD_DIM + 16;
    constexpr int V8T_LD = I8_TILE_TOK + 16;
    constexpr int P8_BYTES = I8_WARPS * 16 * V8T_LD;
    // FP16 V stash: the tile's 32 columns in natural dim order, read back
    // per dim by the requant pass.
    constexpr int FRESH_BYTES = I8_TILE_TOK * HEAD_DIM * 2;
    constexpr int SCRATCH_BYTES = (P8_BYTES > FRESH_BYTES) ? P8_BYTES : FRESH_BYTES;

    constexpr int ALIGN16 = 15;
    // Tile overlay offsets (all 16-aligned).
    constexpr int OFF_K8 = 0;
    constexpr int OFF_KS = (OFF_K8 + I8_TILE_TOK * Q8_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_V8T = (OFF_KS + I8_TILE_TOK * N_WIN * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VS = (OFF_V8T + HEAD_DIM * V8T_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_SCR = (OFF_VS + HEAD_DIM * 2 + ALIGN16) & ~ALIGN16;
    constexpr int TILE_BYTES = OFF_SCR + SCRATCH_BYTES;
    // Prologue overlay (s_q8 only — the Q scales are RESIDENT, below).
    constexpr int PRO_BYTES = I8_M_ROWS * Q8_LD;
    constexpr int ARENA_BYTES = (TILE_BYTES > PRO_BYTES) ? TILE_BYTES : PRO_BYTES;
    // The resident statics beside the arena: Q scales, per-warp slice
    // bindings (palette metadata, rank tables and the map words they were
    // ranked under), per-warp column origins.
    constexpr int RESIDENT_BYTES = I8_M_ROWS * N_WIN * 2
                                 + I8_WARPS * (int)sizeof(WarpPalette<HEAD_DIM>)
                                 + I8_WARPS * 4;
    static_assert(ARENA_BYTES + RESIDENT_BYTES <= i8_smem_budget(HEAD_DIM),
                  "arena + residents must fit the target-residency smem budget");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    // Q scales stay RESIDENT in smem (512 B of the 4-block headroom): the
    // QK fixup reads them as broadcasts, and NOT draining them to
    // registers hands ptxas 4 regs/thread of slack at the 64-reg cap —
    // measured spill traffic was ~25% of global sector volume.
    __shared__ __half s_q_scale[I8_M_ROWS][N_WIN];
    // Per-WARP binding of the slice the warp's current column lives in —
    // palette metadata and rank tables (`WarpPalette`). A warp rebinds it
    // only when its column moves to another slice.
    __shared__ WarpPalette<HEAD_DIM> s_wpal[I8_WARPS];
    // Logical kv position of each warp's first column (column 4w + c sits
    // at s_qpos[w] + c); a warp with no block in the tile publishes a
    // position past every horizon so its columns mask out.
    __shared__ int s_qpos[I8_WARPS];
    constexpr int DEAD_QPOS = 0x7fff0000;

    // Prologue view (dead after the Q drain barrier).
    auto s_q8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena);
    // Tile views (alias the prologue bytes — valid only after the drain).
    auto s_k8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena + OFF_K8);
    auto s_k_scale = reinterpret_cast<__half(*)[N_WIN]>(s_arena + OFF_KS);
    auto s_v8t = reinterpret_cast<int8_t(*)[V8T_LD]>(s_arena + OFF_V8T);
    auto s_v_scale = reinterpret_cast<__half*>(s_arena + OFF_VS);

    // Scratch tenant views (inside the tile overlay). Temporally disjoint:
    //   s_fresh — FP16 V stash for the per-dim requant, STAGING only.
    //   s_p8    — per-warp quantized P tiles, COMPUTE phase only.
    auto s_fresh = reinterpret_cast<__half(*)[HEAD_DIM]>(s_arena + OFF_SCR);
    auto s_p8 = reinterpret_cast<int8_t(*)[16][V8T_LD]>(s_arena + OFF_SCR);

    // ------------------------------------------------------------------
    // Row → (token, head) mapping and per-thread fragment rows.
    // Thread's fragment rows within its warp tile: g = lane>>2 and g+8.
    // ------------------------------------------------------------------
    const int g = lane >> 2;
    const int n0 = (lane & 3) * 2;
    // Row-derived values (token, head, liveness, causal horizon) are
    // recomputed on demand: at the 64-register 4-blocks/SM cap, holding
    // them in arrays is ~10 across-loop registers — guaranteed hot spills.
    auto row_of = [&](int i) { return row_tile * 16 + g + i * 8; };
    auto row_tok = [&](int i) { return t0 + row_of(i) / hpg; };
    auto row_head = [&](int i) {
        int r = row_of(i);
        return first_q_head + (r - (r / hpg) * hpg);
    };
    auto row_live = [&](int i) {
        return (row_of(i) < rows_used) && (row_tok(i) < q_len);
    };

    // ------------------------------------------------------------------
    // Q staging: load, RoPE, per-window int8 quantize (natural dim order).
    // Lane holds dims {lane + 32w : w in 0..N_WIN}; RoPE pair (d, d+HALF)
    // is (w, w + N_WIN/2) — in-thread for HEAD_DIM % 64 == 0.
    // ------------------------------------------------------------------
    //
    // A warp's rows are straight-line code, every row's reads in flight at
    // once: a row past the block's run reads row 0 of the launch (always
    // there) and is zeroed, so no branch stands between one row's reads and
    // the next's. A zero row rotates and quantises to zero codes and a zero
    // scale, exactly what the row would hold unrotated.
    static_assert(I8_M_ROWS % I8_WARPS == 0, "every warp stages the same number of Q rows");
    #pragma unroll
    for (int ri = 0; ri < I8_M_ROWS / I8_WARPS; ++ri) {
        const int r = warp + ri * I8_WARPS;
        int tl = r / hpg;
        int tok = t0 + tl;
        int head = first_q_head + (r - tl * hpg);
        float x[N_WIN];
        bool live = (r < rows_used) && (tok < q_len);
        const QT* qrow = q + (live ? ((int64_t)(q_start + tok) * n_head + head) * HEAD_DIM : 0);
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) {
            const float v = qt_to_f32<QT>(qrow[lane + 32 * w]);
            x[w] = live ? v : 0.f;
        }
        i8_rope_row<N_WIN>(x, prefix_len + tok, lane, rope_interleaved, rope.for_q());
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) {
            float a = fabsf(x[w]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
            float scale = a / 127.f;
            float inv = (scale > 0.f) ? 1.f / scale : 0.f;
            s_q8[r][lane + 32 * w] = i8_quant(x[w], inv);
            if (lane == 0) s_q_scale[r][w] = __float2half(scale);
        }
    }

    // ------------------------------------------------------------------
    // Arena write pre-pass (split 0 only): seal this block's fresh tokens
    // into the writer chunks (unrotated K + Q-capture, straight from the
    // packed inputs — identical semantics to the FP16 kernel's writeback,
    // hoisted out of the tile loop). Writer chunks use the identity palette.
    // ------------------------------------------------------------------
    if (split_idx == 0) {
        const int tok_end = min(t0 + block_m_tok, q_len);
        for (int tok = t0; tok < tok_end; ++tok) {
            int w_slice, w_in_blk;
            resolve_pos(slot_hdr, prefix_len + tok, w_slice, w_in_blk);
            const uint8_t* w_sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, w_slice, n_kv_head);
            const uint8_t* w_head = get_head<HEAD_DIM>(w_sl, kv_head_idx);
            const QT* k_row = k_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            const QT* v_row = v_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            const QT* q_row = q + ((int64_t)(q_start + tok) * n_head + first_q_head) * HEAD_DIM;
            for (int d = tid * 8; d < HEAD_DIM; d += I8_THREADS * 8) {
                int p = d / SUB;
                int local_d = d - p * SUB;
                store_kv_chunk_arena<QT, QT, SUB>(
                    (char*)kvhead_k_ptr<HEAD_DIM>(w_head, p),
                    (char*)kvhead_v_ptr<HEAD_DIM>(w_head, p),
                    &k_row[d], &v_row[d], &q_row[d],
                    kvhead_k_fmt<HEAD_DIM>(w_head, p),
                    kvhead_v_fmt<HEAD_DIM>(w_head, p),
                    0, 0, w_in_blk, local_d, 0, 0);
            }
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Drain Q to registers (block-scope q8-matmul trick): each warp's QK
    // A-operand is its own 16 rows — N_WIN ldmatrix fragments + the two
    // fragment rows' per-window scales. Q never changes across the tile
    // loop, so after this barrier the prologue smem region is dead and
    // the tile overlay owns its bytes.
    // ------------------------------------------------------------------
    uint32_t q_frag[N_WIN][4];
    #pragma unroll
    for (int w = 0; w < N_WIN; ++w) {
        load_a_frag_m16k32_ldmatrix(q_frag[w], &s_q8[row_tile * 16][32 * w], Q8_LD, lane);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Per-warp softmax + output state (registers). PV_H slices = this
    // warp's share of the output dims; they start at
    // dim_part * (HEAD_DIM / DIM_SPLIT).
    // ------------------------------------------------------------------
    constexpr int PV_H = PV_SLICES / DIM_SPLIT;
    const int dim_base = dim_part * (HEAD_DIM / DIM_SPLIT);
    float o_acc[PV_H][4];
    #pragma unroll
    for (int s = 0; s < PV_H; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    // ------------------------------------------------------------------
    // Block walk. The block's query tokens between them select a bounded
    // set of `QB`-position selection blocks whatever the depth; each tile
    // packs the next NB = 32 / QB of them in ascending order, so the tile
    // loop's trip count is bounded by the rows' combined budget. A launch
    // without a selection walks 32-position blocks, one per tile — the
    // dense causal read. The walk is warp-private register state; every
    // warp derives the same block sequence, so the tile's composition
    // stays block-uniform without a handoff (see qsa_walk.cuh).
    //
    // Warp w stages columns 4w..4w+3: the cells `my_to..my_to+3` of the
    // tile's block `my_blk` (one warp per block at QB 4, all eight on the
    // single block at QB 32).
    // ------------------------------------------------------------------
    // A lane slot per 32 query tokens of the block's run: the widest run is
    // I8_M_ROWS tokens (hpg 1).
    constexpr int WALK_ROWS = (I8_M_ROWS + 31) / 32;
    using Walk = QsaWalk<WALK_ROWS>;
    static_assert(I8_M_ROWS <= Walk::MAX_ROWS,
                  "the walk binds one query row per lane slot");
    Walk walk;
    if (qsa_active(sel)) {
        walk.init(sel, q_start + t0, min(block_m_tok, q_len - t0), lane);
    } else {
        walk.init_dense(min(block_m_tok, q_len - t0), I8_TILE_TOK, lane);
    }
    const int QB = walk.ratio;
    const int NB = I8_TILE_TOK / QB;
    const int my_blk = (warp * I8_COLS_PER_WARP) / QB;
    const int my_to = warp * I8_COLS_PER_WARP - my_blk * QB;

    // Per-warp slice binding for sealed columns: the slice whose palette
    // metadata and rank tables sit in s_wpal[warp] (-1 until the first sealed
    // column; the rank tables are rebuilt on the first bind and thereafter
    // only on a map change).
    int bound_slice = -1;

    // ==================================================================
    // Split-KV ownership. A launch of sparse rows gives each split a
    // contiguous RANGE of positions — split s takes the blocks starting in
    // [share_start(s), share_start(s+1)), the s-th equal share of the first
    // row's entry list — and the split seeks straight to it and walks only
    // that. The walk is a serial chain of dependent loads per selected block,
    // so a split that walked the whole selection to keep every
    // `num_splits`-th tile paid the full walk however many splits shared the
    // work: a 5-row verify window at a 2,048-position budget ran ~300 µs per
    // launch, flat across 32 splits. A block belongs to the split its start
    // falls in, so no block is staged twice or skipped across a boundary.
    //
    // A dense row steps from the walk's bound through every position, so a
    // step can straddle a range boundary; a launch with any dense row keeps
    // the round-robin over tiles instead.
    //
    // A run with NO sparse row steps every tile across the 32 positions from
    // where the previous tile stopped, so its t-th tile is the 32 positions
    // from 32·t whatever came before it. Its shards seek straight to their own
    // tiles (`dealt_seek`) instead of walking past every tile dealt to the
    // others — a 294K-token prefix is 9K tiles, of which a shard keeps
    // 1/num_splits. Each shard still owns and packs exactly the tiles the
    // round-robin deals it.
    // ==================================================================
    const bool range_split = num_splits > 1 && !__any_sync(0xffffffffu, walk.dense != 0u);
    const bool dealt_seek = !range_split && num_splits > 1 && walk.all_dense();
    int bound = 0;               // next block start the walk may return
    int split_end = QSA_WALK_END; // first position past this split's range
    if (range_split) {
        bound = walk.share_start(split_idx, num_splits);
        split_end = walk.share_start(split_idx + 1, num_splits);
        walk.seek(bound);
    }

    // ==================================================================
    // Tile loop: each tile is the next NB selected blocks, packed.
    // ==================================================================
    // Visited-tile ordinal, for split-KV round-robin; under `dealt_seek`, the
    // ordinal of this shard's next tile.
    int tile_ord = dealt_seek ? split_idx : 0;
    // The block's causal horizon: no row it serves attends at or past this
    // position. A dense row's walk would otherwise propose every position up
    // to `kv_len` — in a prompt prefilled from zero, four tiles in five of a
    // dense block's walk — each staged, multiplied and then masked to nothing.
    // A tile is skipped only when its FIRST block starts past the horizon: a
    // tile straddling it is kept whole, because the V requant's per-dim scale
    // is taken over every column of the tile, masked or not, and cutting the
    // tile short would move it.
    const int blk_horizon = min(kv_len, prefix_len + min(t0 + block_m_tok, q_len));
    for (;;) {
        if (dealt_seek) bound = tile_ord * I8_TILE_TOK;
        // ---- WALK (every warp, identical): the tile's blocks ----
        // rm[h]: lane's row h's selected-cell bits over the tile's 32
        // columns (block b's cells at bits b·QB ..). my_q: the start of
        // this warp's block, or END when the tile ends before it.
        uint32_t rm[WALK_ROWS];
        #pragma unroll
        for (int h = 0; h < WALK_ROWS; ++h) rm[h] = 0u;
        int my_q = QSA_WALK_END;
        int n_blk = 0;
        for (int b = 0; b < NB; ++b) {
            uint32_t mk[WALK_ROWS];
            int step_end;
            const int q = walk.next(bound, mk, step_end);
            if (q >= kv_len || q >= split_end || (b == 0 && q >= blk_horizon)) break;
            #pragma unroll
            for (int h = 0; h < WALK_ROWS; ++h) rm[h] |= mk[h] << (b * QB);
            if (b == my_blk) my_q = q;
            n_blk = b + 1;
            // The step's own end: through a page layout a block can be shorter
            // than `QB`, and the next block starts where this one ends.
            bound = step_end;
        }
        if (n_blk == 0) break;
        // Round-robin tiles across shards. The skip is block-uniform
        // (every thread computes identical walk state), so the staging
        // barriers below stay convergent.
        const bool mine = range_split || (tile_ord % num_splits) == split_idx;
        tile_ord += dealt_seek ? num_splits : 1;
        if (!mine) continue;

        // -------------------- STAGE (all warps) --------------------
        // Warp w stages its four columns into s_k8 / s_k_scale and stashes
        // V (natural dims, FP16) in s_fresh; the per-dim V requant below
        // runs over the whole tile once every column is in. A column past
        // the tile's blocks (or past kv_len inside a partial last block) is
        // zero and masked with P == 0 in the compute phase; a dead warp's
        // columns start at kv_len, so all four are.
        static_assert(I8_COLS_PER_WARP == 4, "a warp's columns are one token quad");
        const int pos0 = (my_q == QSA_WALK_END) ? kv_len : my_q + my_to;
        const int j0 = warp * I8_COLS_PER_WARP;
        bool from_stage = false;
        if constexpr (PRESTAGED) from_stage = staged && (pos0 & 3) == 0; // warp-uniform
        if (from_stage) {
            // The staged quad at pos0 — what `i8_stage_quad` below would
            // write for these four positions, copied in 16-byte pieces. A
            // column past kv_len was never staged and is zero, as the decode
            // would have made it.
            constexpr int K_CHUNKS = HEAD_DIM / 16;     // 16-byte pieces of a K code row
            constexpr int V_CHUNKS = HEAD_DIM * 2 / 16; // … of an FP16 V row
            constexpr int S_BYTES = N_WIN * 2;          // a K row's window scales
            const int live = kv_len - pos0;             // columns inside the sequence
            const int64_t row0 = (int64_t)kv_head_idx * stage.positions + stage_base + pos0;
            const int8_t* kc = kv_stage_k<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;
            const __half* ks = kv_stage_k_scale<HEAD_DIM>(stage, n_kv_head) + row0 * N_WIN;
            const __half* vv = kv_stage_v<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;
            constexpr int K_PIECES = I8_COLS_PER_WARP * K_CHUNKS;
            constexpr int V_PIECES = I8_COLS_PER_WARP * V_CHUNKS;
            #pragma unroll
            for (int i = 0; i < (K_PIECES + 31) / 32; ++i) {
                const int c = lane + 32 * i;
                if (c < K_PIECES) {
                    const int tt = c / K_CHUNKS;
                    const int off = (c - tt * K_CHUNKS) * 16;
                    int8_t* dst = &s_k8[j0 + tt][off];
                    if (tt < live) i8_cp_async16(dst, kc + (int64_t)tt * HEAD_DIM + off);
                    else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            #pragma unroll
            for (int i = 0; i < (V_PIECES + 31) / 32; ++i) {
                const int c = lane + 32 * i;
                if (c < V_PIECES) {
                    const int tt = c / V_CHUNKS;
                    const int off = (c - tt * V_CHUNKS) * 8; // halves
                    __half* dst = &s_fresh[j0 + tt][off];
                    if (tt < live) i8_cp_async16(dst, vv + (int64_t)tt * HEAD_DIM + off);
                    else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if (lane < I8_COLS_PER_WARP) {
                const int tt = lane;
                __half* dst = &s_k_scale[j0 + tt][0];
                if (tt < live) {
                    kv_stage_cp_async<S_BYTES>(dst, ks + (int64_t)tt * N_WIN);
                } else {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; w += 2)
                        *reinterpret_cast<uint32_t*>(dst + w) = 0u;
                }
            }
        } else {
            i8_stage_quad<QT, HEAD_DIM>(
                s_wpal[warp], bound_slice, slot_hdr, n_kv_head, kv_head_idx,
                k_packed, v_packed, q_start, prefix_len, kv_len, pos0, lane,
                rope_interleaved, rope,
                [&](int tt, int w, int8_t code, float scale) {
                    s_k8[j0 + tt][lane + 32 * w] = code;
                    if (lane == 0) s_k_scale[j0 + tt][w] = __float2half(scale);
                },
                [&](int tt, int w, float v) {
                    s_fresh[j0 + tt][lane + 32 * w] = __float2half(v);
                });
        }
        if (lane == 0) s_qpos[warp] = (my_q == QSA_WALK_END) ? DEAD_QPOS : my_q + my_to;
        if constexpr (PRESTAGED) {
            // cp.async groups are per thread: each drains its own copies
            // before the barrier publishes them (a bare __syncthreads does
            // not fence cp.async).
            i8_cp_commit();
            i8_cp_wait0();
        }
        __syncthreads();
        // V: per natural dim, max-abs over the tile's 32 columns, then
        // requant into the V^T slab (four columns per aligned store).
        for (int rr = tid; rr < HEAD_DIM; rr += I8_THREADS) {
            float a = 0.f;
            #pragma unroll
            for (int j = 0; j < I8_TILE_TOK; ++j)
                a = fmaxf(a, fabsf(__half2float(s_fresh[j][rr])));
            float scale = a / 127.f;
            float inv = (scale > 0.f) ? 1.f / scale : 0.f;
            for (int j4 = 0; j4 < I8_TILE_TOK; j4 += 4) {
                uint32_t pack = 0;
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    pack |= (uint32_t)(uint8_t)i8_quant(
                                __half2float(s_fresh[j4 + jj][rr]), inv)
                            << (8 * jj);
                *(uint32_t*)&s_v8t[rr][j4] = pack;
            }
            s_v_scale[rr] = __float2half(scale);
        }
        __syncthreads();

        // -------------------- COMPUTE (per warp) --------------------
        // QK: 4 column slices × N_WIN window k-steps, FP32 fixup per window.
        //
        // Software-pipelined (the q8-matmul pattern): iteration k+1's
        // fragment + scale smem loads are ISSUED before iteration k's MMA,
        // so their smem-scoreboard latency drains under the tensor-core op
        // and the FP32 fixup instead of stalling the next issue — the
        // profiler's top stall at 33% occupancy. Fully unrolled: the
        // cur/next rotation is register renaming, not copies.
        float sc[4][4]; // [n-slice][fragment c-index]
        #pragma unroll
        for (int s = 0; s < 4; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i) sc[s][i] = 0.f;

        {
            // Q fragments come from the drained registers (q_frag); the Q
            // scales broadcast from resident smem in the fixup. Only the
            // K-side B fragments + scales pipeline through smem.
            uint32_t b_cur[2], b_nxt[2];
            float ks_cur[2], ks_nxt[2];

            load_b_frag_n8k32_ldmatrix(b_cur, &s_k8[0][0], Q8_LD, lane);
            ks_cur[0] = __half2float(s_k_scale[n0][0]);
            ks_cur[1] = __half2float(s_k_scale[n0 + 1][0]);

            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    // Issue iteration k+1's loads first.
                    int it = w * 4 + s;
                    if (it + 1 < N_WIN * 4) {
                        int wn = (it + 1) >> 2;
                        int sn = (it + 1) & 3;
                        load_b_frag_n8k32_ldmatrix(b_nxt, &s_k8[sn * 8][32 * wn], Q8_LD, lane);
                        ks_nxt[0] = __half2float(s_k_scale[sn * 8 + n0][wn]);
                        ks_nxt[1] = __half2float(s_k_scale[sn * 8 + n0 + 1][wn]);
                    }
                    // MMA + fixup on iteration k while the loads fly.
                    int32_t c_i[4] = {0, 0, 0, 0};
                    int32_t d_i[4];
                    mma_int8_m16n8k32(d_i, q_frag[w], b_cur, c_i);
                    float2 qs;
                    qs.x = __half2float(s_q_scale[row_of(0)][w]);
                    qs.y = __half2float(s_q_scale[row_of(1)][w]);
                    sc[s][0] += (float)d_i[0] * qs.x * ks_cur[0];
                    sc[s][1] += (float)d_i[1] * qs.x * ks_cur[1];
                    sc[s][2] += (float)d_i[2] * qs.y * ks_cur[0];
                    sc[s][3] += (float)d_i[3] * qs.y * ks_cur[1];
                    // Rotate (renamed away under full unroll).
                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        b_cur[r] = b_nxt[r];
                        ks_cur[r] = ks_nxt[r];
                    }
                }
            }
        }

        // Mask + scale. Column j's logical kv position is the staging
        // warp's block start plus its cell (s_qpos[j / 4] + j % 4; a dead
        // warp's DEAD_QPOS fails the horizon). The row's selected cells are
        // the walk's bit mask, fetched from the lane that owns the row
        // (row r of the block's run lives on lane r & 31, half r >> 5) —
        // every column a row attends is a bit test, no per-key search. The
        // causal horizons are per-tile transients (dead after this loop).
        int horizon[2];
        uint32_t rmask[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int h = prefix_len + row_tok(i) + 1;
            if (h > kv_len) h = kv_len;
            horizon[i] = row_live(i) ? h : 0;
            const int r = row_tok(i) - t0;
            uint32_t m = __shfl_sync(0xffffffffu, rm[0], r & 31);
            #pragma unroll
            for (int half = 1; half < WALK_ROWS; ++half) {
                const uint32_t mh = __shfl_sync(0xffffffffu, rm[half], r & 31);
                if ((r >> 5) == half) m = mh;
            }
            rmask[i] = row_live(i) ? m : 0u;
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int ja = s * 8 + n0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int j = ja + (i & 1);
                int row = i >> 1;
                const int pos = s_qpos[j / I8_COLS_PER_WARP] + (j % I8_COLS_PER_WARP);
                const bool ok = (pos < horizon[row]) && ((rmask[row] >> j) & 1u);
                sc[s][i] = ok ? sc[s][i] * softmax_scale : -INFINITY;
            }
        }

        // Online softmax per fragment row.
        float m_new[2], alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            float m_tile = -INFINITY;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                m_tile = fmaxf(m_tile, sc[s][row * 2]);
                m_tile = fmaxf(m_tile, sc[s][row * 2 + 1]);
            }
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 2));
            m_new[row] = fmaxf(m_run[row], m_tile);
            alpha[row] = (m_run[row] == -INFINITY) ? 0.f : __expf(m_run[row] - m_new[row]);
        }

        float l_add[2] = { 0.f, 0.f };
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int ja = s * 8 + n0;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                float p0 = (sc[s][row * 2] == -INFINITY || m_new[row] == -INFINITY)
                               ? 0.f
                               : __expf(sc[s][row * 2] - m_new[row]);
                float p1 = (sc[s][row * 2 + 1] == -INFINITY || m_new[row] == -INFINITY)
                               ? 0.f
                               : __expf(sc[s][row * 2 + 1] - m_new[row]);
                l_add[row] += p0;
                l_add[row] += p1;
                // ja is even: the byte pair stores as one aligned u16.
                uint16_t pk = (uint16_t)(uint8_t)(int8_t)rintf(p0 * 127.f) |
                              ((uint16_t)(uint8_t)(int8_t)rintf(p1 * 127.f) << 8);
                *(uint16_t*)&s_p8[warp][g + row * 8][ja] = pk;
            }
        }
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            float ls = l_add[row];
            ls += __shfl_xor_sync(0xffffffffu, ls, 1);
            ls += __shfl_xor_sync(0xffffffffu, ls, 2);
            l_run[row] = l_run[row] * alpha[row] + ls;
            m_run[row] = m_new[row];
        }
        __syncwarp();

        // PV: one m16n8k32 per output-dim slice (k = the tile's 32 tokens),
        // software-pipelined like QK: slice s+1's V fragment + scales issue
        // before slice s's MMA.
        {
            uint32_t pa[4];
            load_a_frag_m16k32_ldmatrix(pa, &s_p8[warp][0][0], V8T_LD, lane);
            uint32_t vb_cur[2], vb_nxt[2];
            float vs_cur[2], vs_nxt[2];
            load_b_frag_n8k32_ldmatrix(vb_cur, &s_v8t[dim_base][0], V8T_LD, lane);
            vs_cur[0] = __half2float(s_v_scale[dim_base + n0]) * (1.f / 127.f);
            vs_cur[1] = __half2float(s_v_scale[dim_base + n0 + 1]) * (1.f / 127.f);
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                if (s + 1 < PV_H) {
                    load_b_frag_n8k32_ldmatrix(vb_nxt, &s_v8t[dim_base + (s + 1) * 8][0], V8T_LD, lane);
                    vs_nxt[0] = __half2float(s_v_scale[dim_base + (s + 1) * 8 + n0]) * (1.f / 127.f);
                    vs_nxt[1] = __half2float(s_v_scale[dim_base + (s + 1) * 8 + n0 + 1]) * (1.f / 127.f);
                }
                int32_t c_i[4] = {0, 0, 0, 0};
                int32_t d_i[4];
                mma_int8_m16n8k32(d_i, pa, vb_cur, c_i);
                o_acc[s][0] = o_acc[s][0] * alpha[0] + (float)d_i[0] * vs_cur[0];
                o_acc[s][1] = o_acc[s][1] * alpha[0] + (float)d_i[1] * vs_cur[1];
                o_acc[s][2] = o_acc[s][2] * alpha[1] + (float)d_i[2] * vs_cur[0];
                o_acc[s][3] = o_acc[s][3] * alpha[1] + (float)d_i[3] * vs_cur[1];
                vb_cur[0] = vb_nxt[0];
                vb_cur[1] = vb_nxt[1];
                vs_cur[0] = vs_nxt[0];
                vs_cur[1] = vs_nxt[1];
            }
        }

        __syncthreads(); // staging buffers are reused next iteration
    }

    // ------------------------------------------------------------------
    // Epilogue. num_splits == 1: normalize and store O directly (natural
    // dims — no permute needed). Otherwise: emit the un-normalized
    // (ΣpV, m, l) partial for this shard; the combine kernel merges.
    // ------------------------------------------------------------------
    if (num_splits == 1) {
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (!row_live(row) || l_run[row] <= 0.f) continue;
            float inv_l = 1.f / l_run[row];
            QT* orow = out + ((int64_t)(q_start + row_tok(row)) * n_head + row_head(row)) * HEAD_DIM;
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                int dim = dim_base + s * 8 + n0;
                orow[dim] = qt_from_f32<QT>(o_acc[s][row * 2] * inv_l);
                orow[dim + 1] = qt_from_f32<QT>(o_acc[s][row * 2 + 1] * inv_l);
            }
        }
    } else {
        constexpr int REC = HEAD_DIM + 2; // [o[HD], m, l]
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (!row_live(row)) continue; // dead rows alias other seqs' rows
            int64_t row_id = (int64_t)(q_start + row_tok(row)) * n_head + row_head(row);
            float* rec = partials + (row_id * num_splits + split_idx) * REC;
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                int dim = dim_base + s * 8 + n0;
                rec[dim] = o_acc[s][row * 2];
                rec[dim + 1] = o_acc[s][row * 2 + 1];
            }
            // Softmax state: every warp of a row-tile group holds identical
            // m/l (duplicated QK); the dim_part==0 warp's lane 0 writes it.
            if (dim_part == 0 && (lane & 3) == 0) {
                rec[HEAD_DIM] = m_run[row];
                rec[HEAD_DIM + 1] = l_run[row];
            }
        }
    }
}

// ============================================================================
// Split-KV combine: one block per output row, merging `num_splits` partials
// with base-e log-sum-exp. Empty shards carry (m = -inf, l = 0) and vanish.
// ============================================================================

// The most shards a launch splits into. A short window over a long prefix —
// a speculative verify of a few rows — runs as few as 4 unsplit blocks, and
// with the walk range-split each shard's latency is its share of the
// selection, so the launch fills every resident slot: 2 blocks/SM at
// HEAD_DIM 256 is 220 slots on 110 SMs, 55 shards of 4 blocks.
constexpr int I8_MAX_SPLITS = 64;

template <typename QT, int HEAD_DIM>
__global__ void paged_prefill_int8_combine_kernel(
    const float* __restrict__ partials,
    QT* __restrict__ out,
    int num_splits,
    int64_t total_rows
) {
    constexpr int REC = HEAD_DIM + 2;
    const int64_t row_id = blockIdx.x;
    if (row_id >= total_rows) return;
    const int d = (int)threadIdx.x;

    __shared__ float s_gm;
    __shared__ float s_scale[I8_MAX_SPLITS]; // exp(m_s - gm) per split
    __shared__ float s_inv_l;

    const float* base = partials + row_id * num_splits * REC;
    // The shards' (m, l) fold on warp 0, two shards per lane: every load is
    // issued at once and the max and the rescaled sum are shuffle trees, so a
    // wide split costs one round trip rather than a serial chain per shard.
    static_assert(I8_MAX_SPLITS <= 64, "the fold gives each lane two shards");
    if (d < 32) {
        const int s0 = d, s1 = d + 32;
        const float m0 = s0 < num_splits ? base[s0 * REC + HEAD_DIM] : -INFINITY;
        const float m1 = s1 < num_splits ? base[s1 * REC + HEAD_DIM] : -INFINITY;
        const float l0 = s0 < num_splits ? base[s0 * REC + HEAD_DIM + 1] : 0.f;
        const float l1 = s1 < num_splits ? base[s1 * REC + HEAD_DIM + 1] : 0.f;
        float gm = fmaxf(m0, m1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, off));
        const float sc0 = (m0 == -INFINITY) ? 0.f : __expf(m0 - gm);
        const float sc1 = (m1 == -INFINITY) ? 0.f : __expf(m1 - gm);
        if (s0 < num_splits) s_scale[s0] = sc0;
        if (s1 < num_splits) s_scale[s1] = sc1;
        float l_tot = l0 * sc0 + l1 * sc1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            l_tot += __shfl_xor_sync(0xffffffffu, l_tot, off);
        if (d == 0) {
            s_gm = gm;
            s_inv_l = (l_tot > 0.f) ? 1.f / l_tot : 0.f;
        }
    }
    __syncthreads();
    if (s_gm == -INFINITY) return; // row never attended anything

    float acc = 0.f;
    // Unrolled so a batch of shards' loads is in flight before their adds.
    #pragma unroll 8
    for (int s = 0; s < num_splits; ++s) {
        acc += base[s * REC + d] * s_scale[s];
    }
    out[row_id * HEAD_DIM + d] = qt_from_f32<QT>(acc * s_inv_l);
}

// ============================================================================
// Launcher
// ============================================================================

template <typename QT, int HEAD_DIM>
inline void launch_paged_prefill_int8(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    void* o_ptr,
    int32_t total_q,
    int32_t batch_size,
    int32_t n_head,
    int32_t n_kv_head,
    int32_t max_q_len,
    float softmax_scale,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    cudaStream_t stream,
    QsaSel sel,
    // The pre-staged K/V (`kv_stage.cuh`): `stage.buf == nullptr` stages
    // nothing and every column is read from its source; otherwise every
    // sequence with `q_len >= stage.min_q_len` is staged into `stage.buf`
    // (`stage_bytes` long) ahead of the kernel, `stage_max_kv` the deepest.
    const PrefillKvStage stage,
    int64_t stage_bytes,
    int32_t stage_max_kv
) {
    // The tile packs 32 / ratio selection blocks, so the ratio must divide
    // the tile. A selection built at any other ratio is a host bug, not a
    // runtime condition — refuse loudly rather than attend the wrong keys.
    if (sel.entries != nullptr && !i8_ratio_supported(sel.ratio)) {
        fprintf(stderr, "PAGED PREFILL INT8: unsupported QSA ratio %d (need 4, 8, 16 or 32)\n",
                sel.ratio);
        abort();
    }
    int hpg = (n_kv_head > 0) ? n_head / n_kv_head : 1;
    if (hpg <= 0) hpg = 1;
    int block_m_tok = i8_m_rows(HEAD_DIM) / hpg;
    if (block_m_tok <= 0) block_m_tok = 1;
    uint32_t grid_x = (uint32_t)((max_q_len + block_m_tok - 1) / block_m_tok);
    if (grid_x == 0) grid_x = 1;

    // Split-KV factor: fan the tile walk across grid.z shards up to this
    // head dim's residency limit. The short-q/long-prefix regime otherwise
    // runs the whole prefix walk in grid_x × n_kv_head × batch blocks — as
    // few as 4 on the production shape — leaving the GPU idle.
    static int s_sm_count = 0;
    if (s_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (s_sm_count <= 0) s_sm_count = 64;
    }
    int base_blocks = (int)grid_x * n_kv_head * (int)batch_size;
    // Split only when the unsplit grid leaves SMs idle — a grid that
    // already covers the SMs loses more to partial-emit + combine traffic
    // than it gains from sharding the walk (measured: q256/prefix2k
    // regressed 2.8 → 3.2 ms when split unconditionally). Fill toward the
    // residency limit but NEVER past it — floor, not ceil: one block over
    // the slot count starts a second wave and near-doubles the makespan
    // (measured: 320 blocks on 304 slots ran 1.69 → 2.36 ms).
    int num_splits = 1;
    if (base_blocks < s_sm_count) {
        num_splits = (i8_min_blocks(HEAD_DIM) * s_sm_count) / base_blocks;
        if (num_splits < 1) num_splits = 1;
        if (num_splits > I8_MAX_SPLITS) num_splits = I8_MAX_SPLITS;
    }

    // Persistent grow-on-demand partial pool (same idiom as the decode
    // split-KV pool: single-stream, grown with `grow_scratch`, which is safe
    // while the thread records a graph and keeps the block it replaces).
    float* partials = nullptr;
    if (num_splits > 1) {
        static void* s_pool = nullptr;
        static size_t s_pool_bytes = 0;
        size_t need = (size_t)total_q * (size_t)n_head * (size_t)num_splits * (HEAD_DIM + 2)
                      * sizeof(float);
        if (grow_scratch(&s_pool, &s_pool_bytes, need) == cudaSuccess) {
            partials = (float*)s_pool;
        }
        if (partials == nullptr) num_splits = 1; // OOM fallback: direct store
    }

    dim3 grid(grid_x, (uint32_t)n_kv_head, (uint32_t)(batch_size * num_splits));
    dim3 block(I8_THREADS, 1, 1);

    // Clear any error left sticky on this thread by a PRIOR launch so the
    // post-launch check below reflects only this kernel — otherwise a stale
    // error is misattributed here (printed against this grid) even though this
    // launch config is valid and its output correct.
    (void)cudaGetLastError();

    if (stage.buf != nullptr) {
        launch_paged_prefill_kv_prestage<QT, HEAD_DIM>(
            k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
            batch_size, n_kv_head, stage_max_kv, rungs, rope_interleaved,
            stage, stage_bytes, stream);
        paged_prefill_int8_kernel<QT, HEAD_DIM, true><<<grid, block, 0, stream>>>(
            (const QT*)q_ptr, (const QT*)k_ptr, (const QT*)v_ptr,
            headers_ptr, cu_seqlens_q, q_lens, kv_lens,
            (QT*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head,
            softmax_scale, rungs, (int)rope_interleaved,
            num_splits, partials, sel, stage);
    } else {
        paged_prefill_int8_kernel<QT, HEAD_DIM, false><<<grid, block, 0, stream>>>(
            (const QT*)q_ptr, (const QT*)k_ptr, (const QT*)v_ptr,
            headers_ptr, cu_seqlens_q, q_lens, kv_lens,
            (QT*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head,
            softmax_scale, rungs, (int)rope_interleaved,
            num_splits, partials, sel, stage);
    }

    if (num_splits > 1) {
        int64_t total_rows = (int64_t)total_q * n_head;
        dim3 cgrid((uint32_t)total_rows);
        dim3 cblock(HEAD_DIM, 1, 1);
        paged_prefill_int8_combine_kernel<QT, HEAD_DIM><<<cgrid, cblock, 0, stream>>>(
            partials, (QT*)o_ptr, num_splits, total_rows);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "PAGED PREFILL INT8 KERNEL LAUNCH FAILED: %s (grid=%d,%d,%d hd=%d splits=%d "
                "staged=%d)\n",
                cudaGetErrorString(err), grid.x, grid.y, grid.z, HEAD_DIM, num_splits,
                (int)(stage.buf != nullptr));
    }
}

/// The pre-staging pass alone, with its own launch check — what a test reads
/// the stage planes back from.
template <typename QT, int HEAD_DIM>
inline void launch_paged_prefill_kv_stage_only(
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    int32_t batch_size,
    int32_t n_kv_head,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    cudaStream_t stream,
    const PrefillKvStage stage,
    int64_t stage_bytes,
    int32_t stage_max_kv
) {
    (void)cudaGetLastError();
    launch_paged_prefill_kv_prestage<QT, HEAD_DIM>(
        k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
        batch_size, n_kv_head, stage_max_kv, rungs, rope_interleaved,
        stage, stage_bytes, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "PAGED PREFILL KV STAGE LAUNCH FAILED: %s (hd=%d max_kv=%d)\n",
                cudaGetErrorString(err), HEAD_DIM, stage_max_kv);
    }
}

} // namespace prefill_int8
