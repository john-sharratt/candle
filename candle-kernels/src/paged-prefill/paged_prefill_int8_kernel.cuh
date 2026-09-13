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
 *
 *  - PER-WARP COLUMN STAGING: warp w owns tile columns 4w..4w+3 and
 *    decodes each straight from its source — a fresh token's packed
 *    input rows, or a sealed token's arena quant blocks / dtype spans
 *    through that token's slice metadata (per-warp palette bases and
 *    rank bytes, rebound only when the column's slice changes). Columns
 *    of one tile may come from different chunks, so the tile carries no
 *    single palette table; every decode is element-wise through the
 *    arena accessors, and K is RoPEd + requantised per (token, window)
 *    while V is stashed as FP16 for a per-dim requant after the barrier.
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
#include "../paged-decode/slot_types.cuh"
#include "../convert/convert_all.cuh"
#include "../mma/mma_wrappers.cuh"
#include "../convert/int8_elem.cuh"
#include "pal_rank.cuh"
#include "kv_store.cuh"
// QSA block-sparse selection — one row per PACKED QUERY (`q_start + token`).
// Null for every model that reads the whole causal prefix.
#include "../qsa_select.cuh"
#include "qsa_walk.cuh"

namespace prefill_int8 {

using fused_attn::load_a_frag_m16k32_ldmatrix;
using fused_attn::load_b_frag_n8k32_ldmatrix;
using fused_attn::mma_int8_m16n8k32;

// Per-element staging/decoding helpers shared with the INT8 tile decode kernel.
using int8_elem::i8_rope_cs;
using int8_elem::i8_apply_rope;
using int8_elem::qt_to_f32;
using int8_elem::qt_from_f32;
using int8_elem::i8_quant;
using int8_elem::i8_arena_elem;

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
template <typename QT, int HEAD_DIM>
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
    const uint32_t* __restrict__ rope_offsets,
    const float* __restrict__ rope_cs,
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
    QsaSel sel
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 prefill: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;       // QK k-step windows (also dims/lane)
    constexpr int PV_SLICES = HEAD_DIM / 8;    // PV n-slices (output dims per mma)
    constexpr int SUB = HEAD_DIM / N_PALETTE;  // palette band width
    // A lane's rank bytes pack (palette, rank) as p<<6 | rank.
    static_assert(N_PALETTE == 4, "rank byte packs the palette into 2 bits");
    static_assert(SUB <= 64, "rank needs 6 bits");
    // Palette maps are compared one 32-bit word per lane.
    constexpr int MAP_WORDS = HEAD_DIM / 16;
    static_assert(MAP_WORDS <= 32, "a palette map must fit one word per lane");

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
    const uint32_t rope_base = rope_offsets[batch_idx];
    const int first_q_head = kv_head_idx * hpg;

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
    // The resident statics beside the arena: Q scales, per-warp palette
    // metadata, per-warp rank tables and the map words they were ranked
    // under, per-warp column origins.
    constexpr int RESIDENT_BYTES = I8_M_ROWS * N_WIN * 2
                                 + I8_WARPS * 2 * N_PALETTE * (8 + 4 + 4 + 4)
                                 + I8_WARPS * 2 * HEAD_DIM
                                 + I8_WARPS * 2 * MAP_WORDS * 4
                                 + I8_WARPS * 4;
    static_assert(ARENA_BYTES + RESIDENT_BYTES <= i8_smem_budget(HEAD_DIM),
                  "arena + residents must fit the target-residency smem budget");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    // Q scales stay RESIDENT in smem (512 B of the 4-block headroom): the
    // QK fixup reads them as broadcasts, and NOT draining them to
    // registers hands ptxas 4 regs/thread of slack at the 64-reg cap —
    // measured spill traffic was ~25% of global sector volume.
    __shared__ __half s_q_scale[I8_M_ROWS][N_WIN];
    // Per-WARP palette extraction metadata for the slice the warp's
    // current column lives in: global decode base, palette scale, format,
    // and quant block bytes (0 ⇒ dtype element addressing). Index [0] = K,
    // [1] = V. Warp-uniform values; reads are smem broadcasts. A warp
    // rebinds them only when its column moves to another slice.
    __shared__ const char* s_wext_base[I8_WARPS][2][N_PALETTE];
    __shared__ float s_wext_scl[I8_WARPS][2][N_PALETTE];
    __shared__ int s_wext_fmt[I8_WARPS][2][N_PALETTE];
    __shared__ int s_wext_bb[I8_WARPS][2][N_PALETTE];
    // Per-warp rank tables — byte (palette << 6 | rank) per natural dim,
    // [0] = K, [1] = V — and the palette-map words they were computed
    // under (rebuilt only when a newly bound slice's maps differ). Lane
    // reads of dims {lane + 32w} touch 32 consecutive bytes: conflict-free.
    __shared__ uint8_t s_wrank[I8_WARPS][2][HEAD_DIM];
    __shared__ uint32_t s_wmap[I8_WARPS][2][MAP_WORDS];
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
    for (int r = warp; r < I8_M_ROWS; r += I8_WARPS) {
        int tl = r / hpg;
        int tok = t0 + tl;
        int head = first_q_head + (r - tl * hpg);
        float x[N_WIN];
        bool live = (r < rows_used) && (tok < q_len);
        if (live) {
            const QT* qrow = q + ((int64_t)(q_start + tok) * n_head + head) * HEAD_DIM;
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) x[w] = qt_to_f32<QT>(qrow[lane + 32 * w]);
            int pos = prefix_len + tok + (int)rope_base;
            i8_apply_rope<HEAD_DIM, N_WIN>(x, pos, lane, rope_interleaved, rope_cs);
        } else {
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) x[w] = 0.f;
        }
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
    static_assert(I8_M_ROWS <= QSA_WALK_MAX_ROWS,
                  "the walk binds one query row per lane slot");
    QsaWalk walk;
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
    // metadata sits in s_wext[warp] and whose rank tables sit in
    // s_wrank[warp] (-1 until the first sealed column; the rank tables
    // are rebuilt on the first bind and thereafter only on a map change).
    int bound_slice = -1;
    auto bind_slice = [&](int sl_idx) {
        const uint8_t* sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, sl_idx, n_kv_head);
        const uint8_t* head = get_head<HEAD_DIM>(sl, kv_head_idx);
        if (lane < 2 * N_PALETTE) {
            const int side = lane / N_PALETTE;
            const int p = lane - side * N_PALETTE;
            const int fmt = side ? kvhead_v_fmt<HEAD_DIM>(head, p)
                                 : kvhead_k_fmt<HEAD_DIM>(head, p);
            const int es = ArenaFormat::float_elem_size(fmt);
            s_wext_base[warp][side][p] = (const char*)(uintptr_t)(
                side ? kvhead_v_ptr<HEAD_DIM>(head, p) : kvhead_k_ptr<HEAD_DIM>(head, p));
            s_wext_fmt[warp][side][p] = fmt;
            s_wext_bb[warp][side][p] = (es == 0) ? ArenaAccessor::get_quant_block_bytes(fmt) : 0;
            s_wext_scl[warp][side][p] = side ? kvhead_v_scale<HEAD_DIM>(head, p)
                                             : kvhead_k_scale<HEAD_DIM>(head, p);
        }
        // Consecutive slices usually share routing: re-rank only when the
        // maps differ from the ones the rank bytes were computed under.
        const uint8_t* k_pal = kvhead_k_pal_map<HEAD_DIM>(head);
        const uint8_t* v_pal = kvhead_v_pal_map<HEAD_DIM>(head);
        const uint32_t kw = (lane < MAP_WORDS) ? ((const uint32_t*)k_pal)[lane] : 0u;
        const uint32_t vw = (lane < MAP_WORDS) ? ((const uint32_t*)v_pal)[lane] : 0u;
        bool same = (bound_slice >= 0);
        if (lane < MAP_WORDS)
            same = same && (kw == s_wmap[warp][0][lane]) && (vw == s_wmap[warp][1][lane]);
        same = __all_sync(0xffffffffu, same);
        if (!same) {
            if (lane < MAP_WORDS) {
                s_wmap[warp][0][lane] = kw;
                s_wmap[warp][1][lane] = vw;
            }
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                int p, rank;
                const int d = lane + 32 * w;
                prefill_pal_rank(k_pal, d, &p, &rank);
                s_wrank[warp][0][d] = (uint8_t)((p << 6) | rank);
                prefill_pal_rank(v_pal, d, &p, &rank);
                s_wrank[warp][1][d] = (uint8_t)((p << 6) | rank);
            }
        }
        bound_slice = sl_idx;
        __syncwarp();
    };

    // ==================================================================
    // Tile loop: each tile is the next NB selected blocks, packed.
    // ==================================================================
    int bound = 0;    // next block start the walk may return
    int tile_ord = 0; // visited-tile ordinal, for split-KV round-robin
    for (;;) {
        // ---- WALK (every warp, identical): the tile's blocks ----
        // rm[h]: lane's row h's selected-cell bits over the tile's 32
        // columns (block b's cells at bits b·QB ..). my_q: the start of
        // this warp's block, or END when the tile ends before it.
        uint32_t rm[QSA_WALK_ROWS_PER_LANE] = { 0u, 0u };
        int my_q = QSA_WALK_END;
        int n_blk = 0;
        for (int b = 0; b < NB; ++b) {
            uint32_t mk[QSA_WALK_ROWS_PER_LANE];
            const int q = walk.next(bound, mk);
            if (q >= kv_len) break;
            rm[0] |= mk[0] << (b * QB);
            rm[1] |= mk[1] << (b * QB);
            if (b == my_blk) my_q = q;
            n_blk = b + 1;
            bound = q + QB;
        }
        if (n_blk == 0) break;
        // Round-robin tiles across shards. The skip is block-uniform
        // (every thread computes identical walk state), so the staging
        // barriers below stay convergent.
        const bool mine = (tile_ord % num_splits) == split_idx;
        tile_ord += 1;
        if (!mine) continue;

        // -------------------- STAGE (all warps) --------------------
        // Warp w decodes its four columns into s_k8 / s_k_scale and stashes
        // V (natural dims, FP16) in s_fresh; the per-dim V requant below
        // runs over the whole tile once every column is in. Sealed columns
        // come from the arena through the warp's slice binding, fresh ones
        // from the packed inputs; a column past the tile's blocks (or past
        // kv_len inside a partial last block) is zero and masked with
        // P == 0 in the compute phase.
        #pragma unroll 1
        for (int tt = 0; tt < I8_COLS_PER_WARP; ++tt) {
            const int j = warp * I8_COLS_PER_WARP + tt;
            const int pos = (my_q == QSA_WALK_END) ? kv_len : my_q + my_to + tt;
            // K stays in registers for RoPE (pairs (w, w + N_WIN/2) are
            // in-thread); V goes straight to the stash, one dim at a time.
            float x[N_WIN];
            if (pos >= kv_len) {
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) {
                    x[w] = 0.f;
                    s_fresh[j][lane + 32 * w] = __float2half(0.f);
                }
            } else if (pos >= prefix_len) {
                const int tok = pos - prefix_len; // fresh token index
                const QT* kr = k_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
                const QT* vr = v_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) {
                    x[w] = qt_to_f32<QT>(kr[lane + 32 * w]);
                    s_fresh[j][lane + 32 * w] = __float2half(qt_to_f32<QT>(vr[lane + 32 * w]));
                }
            } else {
                int sl_idx, in_blk;
                resolve_pos(slot_hdr, pos, sl_idx, in_blk);
                if (sl_idx != bound_slice) bind_slice(sl_idx); // warp-uniform
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) {
                    const int d = lane + 32 * w;
                    const int tk = s_wrank[warp][0][d];
                    const int pk = (tk >> 6) & (N_PALETTE - 1);
                    x[w] = i8_arena_elem(s_wext_fmt[warp][0][pk], s_wext_bb[warp][0][pk],
                                         s_wext_base[warp][0][pk], tk & 63, in_blk,
                                         s_wext_scl[warp][0][pk], SUB);
                    const int tv = s_wrank[warp][1][d];
                    const int pv = (tv >> 6) & (N_PALETTE - 1);
                    const float v = i8_arena_elem(s_wext_fmt[warp][1][pv], s_wext_bb[warp][1][pv],
                                                  s_wext_base[warp][1][pv], tv & 63, in_blk,
                                                  s_wext_scl[warp][1][pv], SUB);
                    s_fresh[j][lane + 32 * w] = __float2half(v);
                }
            }
            if (pos < kv_len)
                i8_apply_rope<HEAD_DIM, N_WIN>(x, pos + (int)rope_base, lane, rope_interleaved, rope_cs);
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                float a = fabsf(x[w]);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                float scale = a / 127.f;
                float inv = (scale > 0.f) ? 1.f / scale : 0.f;
                s_k8[j][lane + 32 * w] = i8_quant(x[w], inv);
                if (lane == 0) s_k_scale[j][w] = __float2half(scale);
            }
        }
        if (lane == 0) s_qpos[warp] = (my_q == QSA_WALK_END) ? DEAD_QPOS : my_q + my_to;
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
            const uint32_t m0 = __shfl_sync(0xffffffffu, rm[0], r & 31);
            const uint32_t m1 = __shfl_sync(0xffffffffu, rm[1], r & 31);
            rmask[i] = row_live(i) ? ((r >= 32) ? m1 : m0) : 0u;
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
    __shared__ float s_scale[32]; // exp(m_s - gm) per split (num_splits ≤ 32)
    __shared__ float s_inv_l;

    const float* base = partials + row_id * num_splits * REC;
    if (d == 0) {
        float gm = -INFINITY;
        for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, base[s * REC + HEAD_DIM]);
        float l_tot = 0.f;
        for (int s = 0; s < num_splits; ++s) {
            float m_s = base[s * REC + HEAD_DIM];
            float sc = (m_s == -INFINITY) ? 0.f : __expf(m_s - gm);
            s_scale[s] = sc;
            l_tot += base[s * REC + HEAD_DIM + 1] * sc;
        }
        s_gm = gm;
        s_inv_l = (l_tot > 0.f) ? 1.f / l_tot : 0.f;
    }
    __syncthreads();
    if (s_gm == -INFINITY) return; // row never attended anything

    float acc = 0.f;
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
    const uint32_t* rope_offsets,
    const float* rope_cs,
    int32_t rope_interleaved,
    cudaStream_t stream,
    QsaSel sel = {nullptr, nullptr, nullptr, nullptr, 0, 1}
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
        if (num_splits > 32) num_splits = 32;
    }

    // Persistent grow-on-demand partial pool (same idiom as the decode
    // split-KV pool: single-stream, freed only on growth).
    float* partials = nullptr;
    if (num_splits > 1) {
        static float* s_pool = nullptr;
        static size_t s_pool_elems = 0;
        size_t need = (size_t)total_q * (size_t)n_head * (size_t)num_splits * (HEAD_DIM + 2);
        if (need > s_pool_elems) {
            if (s_pool != nullptr) {
                // Drain the stream before freeing: cudaFree is not
                // stream-ordered, and an earlier split launch on this stream
                // may still be writing the old pool. Growth is rare (a new
                // high-water total_q), so the sync cost is amortized away.
                cudaStreamSynchronize(stream);
                cudaFree(s_pool);
            }
            if (cudaMalloc(&s_pool, need * sizeof(float)) != cudaSuccess) {
                s_pool = nullptr;
                s_pool_elems = 0;
            } else {
                s_pool_elems = need;
            }
        }
        partials = s_pool;
        if (partials == nullptr) num_splits = 1; // OOM fallback: direct store
    }

    dim3 grid(grid_x, (uint32_t)n_kv_head, (uint32_t)(batch_size * num_splits));
    dim3 block(I8_THREADS, 1, 1);

    // Clear any error left sticky on this thread by a PRIOR launch so the
    // post-launch check below reflects only this kernel — otherwise a stale
    // error is misattributed here (printed against this grid) even though this
    // launch config is valid and its output correct.
    (void)cudaGetLastError();

    paged_prefill_int8_kernel<QT, HEAD_DIM><<<grid, block, 0, stream>>>(
        (const QT*)q_ptr, (const QT*)k_ptr, (const QT*)v_ptr,
        headers_ptr, cu_seqlens_q, q_lens, kv_lens,
        (QT*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head,
        softmax_scale, rope_offsets, rope_cs, (int)rope_interleaved,
        num_splits, partials, sel);

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
                "PAGED PREFILL INT8 KERNEL LAUNCH FAILED: %s (grid=%d,%d,%d hd=%d splits=%d)\n",
                cudaGetErrorString(err), grid.x, grid.y, grid.z, HEAD_DIM, num_splits);
    }
}

} // namespace prefill_int8
