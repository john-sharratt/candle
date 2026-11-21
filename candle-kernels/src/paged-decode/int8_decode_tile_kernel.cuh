#pragma once
// ============================================================================
// INT8 TILE DECODE — compacted 32-token tiles on the INT8 tensor cores.
//
// The decode kernel for wide heads (HEAD_DIM 256 — Qwen3.8-Flash-Next's
// full-attention layers: 24 query heads over 2 KV heads, 12 per group). A
// tile is 32 columns = 8 GROUPS of 4 consecutive tokens of one slice, and
// the groups of a tile are independent: each names its own slice, its own
// 4-token run inside that slice, and the 4-bit mask of the run's live
// tokens. A block decodes each tile's K and V straight from the arenas
// into int8 tiles and runs the group's query heads against them as
// m16n8k32 INT8 MMAs. The head group fills 12 of the MMA's 16 rows; the
// rest are zero and cost nothing (64 MMAs per tile for the whole block).
//
// The tile count is what makes the kernel flat in depth. Two walks feed the
// same tile loop, both block-uniform:
//
//   dense   — split s takes slices [s·per, (s+1)·per) of the slot; a tile is
//             one whole slice, its 8 groups the slice's 8 quads with the
//             slice's live-token mask.
//   sparse  — under QSA the row attends a selected set of positions; split s
//             takes INT8_TILE_ENTRIES_PER_SPLIT of the row's ascending entry
//             list, warp 7 resolves each entry's ≤ 4 consecutive cells to
//             (slice, within) and packs every maximal run of consecutive
//             cells into ONE group. Groups fill tiles 8 at a time, so a
//             split's tile count is a function of its entry count alone —
//             at 128K a 512-entry row is 64 tiles wherever its entries
//             fall, where a one-slice-per-tile walk would be 512.
//
// Per tile the block holds one descriptor per (group, side, palette): the
// palette's arena pointer, format and scale, plus the group's two palette
// maps. They are read a tile AHEAD: while tile t
// computes, 64 threads walk tile t+1's slice → head → palette chain into
// registers (three dependent loads that would otherwise head the tile's
// critical path), issue L2 prefetches for the spans they resolve, and
// commit the registers to the other half of a double-buffered descriptor
// table at the end of the tile; every thread likewise carries one word of
// the next tile's palette maps. From the map words the per-group palette
// rank tables are rebuilt cooperatively — 16 lanes per (group, side), one
// 16-dim word each, a segmented popcount prefix — and only for a group
// whose map actually changed from the tile before.
//
// The prologue overlaps its own chains the same way: while warps 0–6 stage
// Q and warp 6 scatters the new token, warp 7 resolves the split's first
// selection chunk (sparse) or builds the window table (dense); one barrier
// publishes those, then every thread stages its own item of tile 0 under
// the tile loop's roles and a second barrier publishes the tile.
//
// Decoding is quad-wide: a thread reads FOUR consecutive dims at a time,
// and a quad that is four consecutive ranks of one dtype palette is one
// aligned vector load per token (8 bytes of bf16) rather than four
// strided scalars. K: warp = group, lane = quad — a lane's eight dims are
// its four RoPE pairs, the cos/sin of all four frequencies two float4s
// per token, the rotation in-register, and a warp's quads tile whole
// 32-dim scale windows so the per-token window absmax is a segmented
// shuffle. V: warp = 32-dim window, lane = (token quarter, quad) — a
// lane's 4 dims × 8 tokens, the per-dim tile absmax an in-register max
// plus two shuffles. Both issue every load of a quad's group under ONE
// format dispatch so the latencies overlap, then quantise four values
// into one word with the saturating int8 pack. Only live tokens are
// loaded — a sparse group touches 4 positions and pays for 4.
//
// That vector path is the float-chunk case: a band map over narrow dtype
// palettes. A quantised or channel-mapped group decodes by the BLOCK
// path instead — slot-wise, a warp taking a whole palette's ranks under
// one warp-uniform format dispatch, the slot's dim from the group's
// inverse rank table — so no lane waits on another lane's format body.
// K stages the decoded group by dim in shared memory and reads it back
// by quad; V gathers the per-dim tile absmax through a shared atomic max
// and quantises straight into the transposed slab. The choice is per
// warp on K (a warp is a group) and for the whole V side at once (the
// two paths partition the head differently).
//
// Masked tokens: a masked K column is quantised from whatever its arena
// bytes hold — K's scale is per token, so nothing crosses into another
// column, and its score is forced to -inf before the softmax. A masked V
// token reads as 0 (V's scale is per dim across the tile) and gets P = 0
// in the int8 P tile.
//
// QK and the softmax run once per tile on warps 0–3, a warp per 8 columns
// over the whole head, so each lane's scores land in the cells its softmax
// reads and no score crosses shared memory; the other four warps carry
// only V. The softmax is in base 2 with the softmax scale folded into the
// Q window scales; the P tile, the row rescale factors and the partial
// row sums cross to the PV warps through shared memory. Every warp issues
// its V loads before the softmax warps start QK and holds only the raw
// words (two per token) until after the softmax, so the V latency hides
// under the MMAs and exp2s rather than heading its own phase.
//
// Each split emits one un-normalised (ΣpV, m, l) partial per query head into
// the decode partial pool; `int8_decode_combine_kernel` merges the splits. A
// split with no tokens writes only (m = -1e38, l = 0) — the combine skips a
// partial on l == 0 without reading its accumulator.
//
// Residency: 8 warps × 128 registers, ~44.8 KB static smem → 2 blocks per
// SM (`TILE_MIN_BLOCKS`). Three blocks at 80 registers was measured slower
// at every depth: the kernel is bound by each warp's own dependent chain,
// and the spills that 80 registers force lengthen it more than the third
// block's warps shorten it.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "../arena_table.cuh"
#include "../blocks.cuh"
#include "../convert/convert_all.cuh"
#include "../convert/int8_elem.cuh"
#include "../mma/mma_wrappers.cuh"
#include "../qsa_select.cuh"
#include "slot_types.cuh"
#include "decode_helpers.cuh"

namespace fused_attn {

#include "int8_decode_tile_kernel/tile_descriptors.cuh"

#include "int8_decode_tile_kernel/quad_decode.cuh"

#include "int8_decode_tile_kernel/k_v_decode.cuh"

#include "int8_decode_tile_kernel/fused_kernel.cuh"

} // namespace fused_attn
