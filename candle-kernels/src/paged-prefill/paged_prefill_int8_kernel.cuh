/*
 * ============================================================================
 * INT8 PREFIX-ATTENTION PREFILL KERNEL
 * ============================================================================
 *
 * Causal prefill attention over a
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
 *  - A BOUNDED CUT (`PrefillCut`): the launcher runs the grid in ROW GROUPS
 *    — contiguous ranges of grid-x blocks, every sequence of the batch — and
 *    each row group in KEY CHUNKS: one key window per launch, staged ahead of
 *    it, so the stage holds one window per staged sequence whatever the
 *    depth. A block whose walk reaches the window's end stops at a TILE
 *    boundary (never inside one: the V requant's scale is per tile), leaves
 *    its FP32 online-softmax state and its walk bound in the carry, and the
 *    next window's launch reloads both and re-walks from that bound — the
 *    same tiles, the same arithmetic, in the same order, so the result is the
 *    uncut launch's bit for bit. Columns of a tile past the window decode
 *    from their source, which is the same bytes the stage would hold. A
 *    sequence too short to stage runs whole in its group's first window.
 *
 *  - FRESH TOKENS FROM THE INPUTS: the q_len new tokens are staged straight
 *    from the packed q/k/v tensors (never read back from the arena); the
 *    arena write of their K/V is an independent pre-pass (z == 0 of a row
 *    group's first key window only).
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

#include "paged_prefill_int8_kernel/prefill_helpers.cuh"

#include "paged_prefill_int8_kernel/prefill_kernel.cuh"

#include "paged_prefill_int8_kernel/combine_and_launch.cuh"

} // namespace prefill_int8
