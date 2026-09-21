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

using int8_elem::i8_apply_rope;
using int8_elem::i8_quant;
using int8_elem::i8_pack2;
using int8_elem::i8_pack4;
using int8_elem::i8_with_format;
using int8_elem::i8_with_dtype_format;
using int8_elem::i8_is_dtype_format;
using int8_elem::i8_with_narrow_dtype_format;
using int8_elem::i8_is_narrow_dtype_format;
using int8_elem::i8_dtype_cvt4;
using int8_elem::i8_tag_quad;
using int8_elem::I8IsDtypeTag;
using int8_elem::I8DtypeTag;
using int8_elem::I8BlockQuad;

constexpr int TILE_WARPS = 8;
constexpr int TILE_THREADS = TILE_WARPS * WARP_SIZE;
constexpr int TILE_TOK = CHUNK_SIZE;          // columns per tile
constexpr int TILE_M_ROWS = 16;               // MMA rows: the group's query heads
constexpr int TILE_MIN_BLOCKS = 2;            // residency target (128 regs/thread)
/// Warps that run the softmax: warp ns owns the tile's columns ns·8..+7.
constexpr int TILE_SM_WARPS = TILE_TOK / 8;
/// The other warps: they read and quantise V while the softmax warps run
/// QK, each owning HEAD_DIM / TILE_V_WARPS dims as windows of 32.
constexpr int TILE_V_WARPS = TILE_WARPS - TILE_SM_WARPS;
/// Named barrier the softmax warps meet at between their row-max post and
/// its read; barrier 0 is `__syncthreads`.
constexpr int TILE_SM_BARRIER = 1;
/// Named barrier publishing the K slab: every warp writes its own group's
/// four rows and a softmax warp's eight columns come from two of them, so
/// all eight warps arrive once their rows are stored and only the softmax
/// warps wait — the V warps go straight on to their loads.
constexpr int TILE_K_BARRIER = 2;
/// Named barrier the V warps meet at on the block path, between their
/// tile-absmax atomics and the quantise that reads the result; the
/// softmax warps never touch it.
constexpr int TILE_VABS_BARRIER = 3;
/// The softmax runs in base 2: the Q scale carries softmax_scale · log2(e),
/// so a score is already a log2-domain logit and exp2 is one MUFU. The
/// running max goes out to the combine in natural-log units (× ln 2).
constexpr float TILE_LOG2E = 1.4426950408889634f;
constexpr float TILE_LN2 = 0.6931471805599453f;
/// Tokens per group: one selection entry's cell count, so an entry's run of
/// consecutive positions is at most one group.
constexpr int GROUP_TOK = 1 << QSA_CELL_BITS;
constexpr int TILE_GROUPS = TILE_TOK / GROUP_TOK;
/// Group SLOTS a tile carries: the eight quads a whole window is, plus one.
/// A window off the physical 4-grid — every window after a section sealed
/// mid-quad, until the next boundary — spans NINE aligned quads (a partial
/// one at each edge), and the ninth rides in slot TILE_GROUPS: staged,
/// tabled and decoded by warp 0 as a second round for K, and by the V warps
/// as one extra token per lane-octet. A window of more than TILE_SLOTS
/// quads (two holes inside it) is still taken in passes of TILE_GROUPS.
constexpr int TILE_SLOTS = TILE_GROUPS + 1;
/// Selection entries per split under QSA. A split's tiles hold its entries'
/// runs 8 groups per tile, so 8 entries is one full tile when every entry
/// is a single run; the released checkpoint's ≤ 514-entry rows spread over
/// ~65 splits, ~130 blocks with 2 kv heads.
constexpr int INT8_TILE_ENTRIES_PER_SPLIT = 8;

/// A group descriptor: `slice << 9 | within << 4 | mask`. `within` is the
/// slice-local index of the group's first token — always a multiple of 4,
/// so a group is one aligned token quad of every dim's block — and `mask`
/// bit j marks token within + j live. A zero mask is a dead group (nothing
/// is read for it).
__device__ __forceinline__ uint32_t grp_pack(int slice, int within, uint32_t mask) {
    return ((uint32_t)slice << 9) | ((uint32_t)within << 4) | mask;
}
__device__ __forceinline__ uint32_t grp_mask(uint32_t d) { return d & 15u; }
__device__ __forceinline__ int grp_within(uint32_t d) { return (int)((d >> 4) & 31u); }
__device__ __forceinline__ int grp_slice(uint32_t d) { return (int)(d >> 9); }

/// Per-(group, side, palette) extraction metadata of one tile: the
/// palette's arena base, palette scale and format. Index [g][0] = K,
/// [g][1] = V.
struct TileExt {
    const char* gbase[TILE_SLOTS][2][N_PALETTE];
    float scl[TILE_SLOTS][2][N_PALETTE];
    uint8_t fmt[TILE_SLOTS][2][N_PALETTE];
};

/// The tile's groups: descriptor, the logical position of the group's
/// `within` token (so token j of the group sits at position rope0[g] + j),
/// and the tile COLUMN that token lands on.
///
/// **A tile is a logical window, and a column is a logical position.**
/// Tile t of a dense split covers positions [32t, 32t + 32); column c of it
/// IS position 32t + c, whichever slice — and whichever physical quad of it
/// — holds that token. A group is still one aligned physical quad of one
/// slice (the loaders, the rank tables and the palette metadata are all per
/// slice, and the block path asserts the alignment), so `col0` is what says
/// where the quad's four rows go: token j of group g is column col0[g] + j,
/// and the group's mask has a bit only for tokens inside the window. A quad
/// straddling the window's edge appears in both windows, each time with the
/// columns outside that window masked off, so col0 can be as low as
/// −(GROUP_TOK − 1). A dead group (mask 0) has no columns at all.
///
/// This is what makes the attention a function of the tokens rather than of
/// their chunking: the softmax's running max advances at logical window
/// boundaries, the int8 codes it produces are taken against a reference that
/// depends only on positions, and two slots holding the same tokens at the
/// same positions in different chunk layouts produce the same bytes. Tiling
/// by physical slice instead — the previous rule, `tok = 4·g + j` — put the
/// reference at slice boundaries, so a hole (a section sealed mid-chunk, a
/// tombstoned turn, a skipped thinking block) moved every later token's
/// reference and changed the output.
struct TileGroups {
    uint32_t desc[TILE_SLOTS];
    int rope0[TILE_SLOTS];
    int8_t col0[TILE_SLOTS];
    /// Quads the tile's window spans. Up to TILE_SLOTS they fit one pass (the
    /// ninth in slot TILE_GROUPS); above that the tile is taken in passes of
    /// TILE_GROUPS groups that share one softmax and one V scale.
    int n_groups;
};

/// What a thread carries from staging the NEXT tile to committing it:
/// its palette-map word (every thread), and for the 64 descriptor
/// stagers the group descriptor plus the palette's pointer, scale and
/// format. `stage_commit` writes it all to the tile's descriptor buffer.
struct NextTile {
    uint32_t map;
    uint32_t desc;
    int rope0;
    int col0;
    int ngroups;
    uint64_t ptr;
    float scl;
    int fmt;
};

/// The window's column mask from a group: its live bits shifted to its
/// columns. A negative `col0` is a quad straddling the window's start, whose
/// leading tokens belong to the previous window and are not in the mask.
__device__ __forceinline__ uint32_t grp_cols(uint32_t mask, int col0) {
    return col0 >= 0 ? (mask << col0) : (mask >> (-col0));
}

/// Byte offset of (row, col) in an MMA slab of ROW_BYTES-byte rows with no
/// pad: the row's 16-byte chunks are XOR-swizzled by a per-row key so the
/// eight rows an ldmatrix phase reads at one logical chunk land in eight
/// distinct bank quads. A 256-byte row spans all 32 banks itself, so the
/// key is row & 7; a 32-byte row shares its banks with the three rows
/// beside it, so the key is (row >> 2) & 1 — rows 0-3 straight, rows 4-7
/// with their two chunks swapped. A value narrower than a chunk keeps its
/// place inside it, so word and half-word stores address through this
/// too.
template <int ROW_BYTES>
__device__ __forceinline__ int sw_off(int row, int col) {
    constexpr int CHUNKS = ROW_BYTES / 16;
    constexpr int ROWS_PER_128 = ROW_BYTES < 128 ? 128 / ROW_BYTES : 1;
    constexpr int KEY = (CHUNKS < 8 ? CHUNKS : 8) - 1;
    static_assert(ROW_BYTES % 16 == 0 && CHUNKS >= 2, "a swizzled row is whole chunks");
    return row * ROW_BYTES + (((col >> 4) ^ ((row / ROWS_PER_128) & KEY)) << 4) + (col & 15);
}

/// Shared address of the lane's ldmatrix row for a 16×32 int8 A fragment
/// (x4: tile t>>3 is rows (t>>3 & 1)·8.., K half t>>4) or an 8×32 int8 B
/// fragment (x2: K half (t>>3) & 1) of a swizzled slab, at column col0.
template <int ROW_BYTES>
__device__ __forceinline__ uint32_t sw_a_frag_addr(const int8_t* slab, int col0, int lane) {
    const int row = ((lane >> 3) & 1) * 8 + (lane & 7);
    return static_cast<uint32_t>(__cvta_generic_to_shared(
        slab + sw_off<ROW_BYTES>(row, col0 + (lane >> 4) * 16)));
}
template <int ROW_BYTES>
__device__ __forceinline__ uint32_t sw_b_frag_addr(const int8_t* slab, int row0, int col0, int lane) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(
        slab + sw_off<ROW_BYTES>(row0 + (lane & 7), col0 + ((lane >> 3) & 1) * 16)));
}

/// Valid tokens of slice `s` with header `h`: its stored length, plus one
/// for the write slice's freshly scattered token when that token landed
/// inside the slice.
__device__ __forceinline__ int tile_slice_eff_len(const TokenSliceHdr& h, int s, int write_slice_idx) {
    int len = h.len();
    if (s == write_slice_idx && len < CHUNK_SIZE && h.offset() + len < CHUNK_SIZE) len += 1;
    return len;
}

/// What the resolve publishes per group beside its descriptor: the slice's
/// KvHead record pointer and `rope − offset` (so a token at `within` sits
/// at position rope_base + within). The stager reads these instead of the
/// slice header, which takes the header's global round off its chain.
struct GroupSrc {
    uint64_t kvheads;
    int rope_base;
};

/// Sparse walk (one warp): resolve this split's selection entries into
/// groups. Lane `l` holds entry e_lo + l as `ent` (`has` when it exists —
/// the caller issues the load, so it is in flight beside the slot header
/// rather than behind it), finds the slice holding its first position
/// (rope ranges ascend and do not overlap) by a guess-then-gallop search,
/// then walks the entry's ≤ 4 consecutive cells with a forward cursor. A
/// cell no slice holds — a selection built against a different cache than
/// the one being read — is dropped: attending a slot outside its slice
/// would read another token's K/V.
///
/// The walk reads single header fields and holds only the cursor: this
/// resolve also runs mid-loop, when a chunk ends with the tile loop's
/// state live, and every register it holds there is one the loop loses.
/// The `GroupSrc` for each run start is read whole ([`TokenSliceHdr`])
/// at publish, from a line the walk just touched.
///
/// Every maximal run of cells at consecutive `within` of one slice, inside
/// one 4-aligned quad of the slice, is one group: the group's `within` is
/// the quad's first token and its mask is the run's tokens at their place
/// in the quad, so a slice whose tokens sit off the cell grid costs an
/// extra group with dead columns rather than an unaligned block read. A
/// warp scan numbers the runs so the groups come out packed in ascending
/// position order. Groups past the last are dead (zero).
/// The last slice whose rope base is at or below `pos` — the slice holding
/// position `pos` when one does (a slice's tokens run from its rope base for
/// its length; an empty slice at the same base sits after the one that holds
/// the position and is never the answer here, because its base is not
/// BELOW the position it follows).
///
/// Slices fill in position order at CHUNK_SIZE tokens each, so the slice
/// holding `pos` is almost always pos / CHUNK_SIZE: probe it and its successor
/// with one pair of loads issued together (the successor's rope is what closes
/// the bracket on a hit), gallop from there in whichever direction the guess
/// missed, and binary-search only inside the bracket the gallop closed.
template <int HEAD_DIM>
__device__ __forceinline__ int tile_slice_holding(
    uint64_t slices_ptr, int n_slices, int n_kv_head, int pos)
{
    auto rope_of = [&](int i) {
        return (int)slice_rope(get_slice<HEAD_DIM>(slices_ptr, i, n_kv_head));
    };
    const int g = min(max(pos / CHUNK_SIZE, 0), n_slices - 1);
    const int rg = rope_of(g);
    const int rg1 = rope_of(min(g + 1, n_slices - 1));
    int lo_s, hi_s;
    if (rg <= pos) {
        lo_s = g;
        hi_s = g;
        if (g + 1 < n_slices && rg1 <= pos) {
            lo_s = g + 1;
            int probe = g + 2, step = 2;
            while (probe < n_slices && rope_of(probe) <= pos) {
                lo_s = probe;
                probe = lo_s + step;
                step <<= 1;
            }
            hi_s = min(probe, n_slices) - 1;
        }
    } else {
        hi_s = g - 1;
        int probe = g - 1, step = 1;
        while (probe > 0 && rope_of(probe) > pos) {
            hi_s = probe - 1;
            probe -= step;
            step <<= 1;
        }
        lo_s = max(probe, 0);
    }
    while (lo_s < hi_s) {
        const int mid = (lo_s + hi_s + 1) >> 1;
        if (rope_of(mid) <= pos) lo_s = mid; else hi_s = mid - 1;
    }
    return lo_s;
}

template <int HEAD_DIM>
__device__ __forceinline__ void tile_resolve_entries(
    uint32_t ent, bool has, const QsaSel& sel, int row,
    uint64_t slices_ptr, int n_slices, int write_slice_idx, int n_kv_head,
    int lane, uint32_t* s_grp, GroupSrc* s_grp_src, int* s_n_tiles)
{
    constexpr int MAX_GROUPS = INT8_TILE_ENTRIES_PER_SPLIT * GROUP_TOK;
    constexpr int NO_SLICE = -1;
    for (int i = lane; i < MAX_GROUPS; i += WARP_SIZE) s_grp[i] = 0u;
    __syncwarp();

    // The entry's start and width come through the row's page layout: a
    // projected prefix is pages whose last blocks are short, so every block
    // behind the first page starts somewhere other than `block * ratio` (see
    // `qsa_block_width_from`).
    const uint32_t blk = ent >> QSA_CELL_BITS;
    const int pos0 = has ? qsa_block_start(sel, row, blk) : 0;
    const int cells = has ? max(0, min((int)(ent & ((1u << QSA_CELL_BITS) - 1u)) + 1,
                                       qsa_block_width_from(sel, row, blk, pos0)))
                          : 0;
    auto slice_at = [&](int i) { return get_slice<HEAD_DIM>(slices_ptr, i, n_kv_head); };
    auto rope_of = [&](int i) { return (int)slice_rope(slice_at(i)); };

    int cs[GROUP_TOK];
    int cw[GROUP_TOK];
    int s = 0;
    if (has) s = tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, pos0);
    #pragma unroll
    for (int c = 0; c < GROUP_TOK; ++c) {
        cs[c] = NO_SLICE;
        cw[c] = 0;
        if (c < cells) {
            const int pos = pos0 + c;
            while (s + 1 < n_slices && rope_of(s + 1) <= pos) ++s;
            const uint8_t* sp = slice_at(s);
            const int local = pos - (int)slice_rope(sp);
            const int off = (int)slice_offset(sp);
            int eff = (int)slice_len(sp);
            if (s == write_slice_idx && eff < CHUNK_SIZE && off + eff < CHUNK_SIZE) eff += 1;
            if (local >= 0 && local < eff) {
                cs[c] = s;
                cw[c] = off + local;
            }
        }
    }

    // A cell starts a run unless it is the next `within` of the previous
    // cell's slice inside the same aligned quad; `len[c]` is the run
    // length from cell c onward.
    bool start[GROUP_TOK];
    int len[GROUP_TOK];
    #pragma unroll
    for (int c = 0; c < GROUP_TOK; ++c) {
        start[c] = cs[c] != NO_SLICE &&
                   (c == 0 || cs[c - 1] != cs[c] || cw[c - 1] + 1 != cw[c] || (cw[c] & 3) == 0);
    }
    #pragma unroll
    for (int c = GROUP_TOK - 1; c >= 0; --c) {
        len[c] = 1;
        if (c + 1 < GROUP_TOK && cs[c + 1] != NO_SLICE && !start[c + 1]) len[c] = len[c + 1] + 1;
    }
    int starts = 0;
    #pragma unroll
    for (int c = 0; c < GROUP_TOK; ++c) starts += start[c] ? 1 : 0;
    int incl = starts;
    #pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, incl, d);
        if (lane >= d) incl += v;
    }
    int r = incl - starts;
    #pragma unroll
    for (int c = 0; c < GROUP_TOK; ++c) {
        if (start[c]) {
            const TokenSliceHdr h = load_token_slice<HEAD_DIM>(slices_ptr, cs[c], n_kv_head);
            s_grp[r] = grp_pack(cs[c], cw[c] & ~3, ((1u << len[c]) - 1u) << (cw[c] & 3));
            s_grp_src[r].kvheads = h.kvheads_ptr();
            s_grp_src[r].rope_base = h.rope_base() - h.offset();
            ++r;
        }
    }
    const int total = __shfl_sync(0xffffffffu, incl, WARP_SIZE - 1);
    if (lane == 0) *s_n_tiles = (total + TILE_GROUPS - 1) / TILE_GROUPS;
}

/// What a lane knows about one quad of one side before decoding it: the
/// rank-table word covering its four dims, the group's live mask and
/// first token, the palette format of its first dim, and whether the four
/// ranks are consecutive from a multiple of 4 inside ONE palette (`vec`:
/// a dtype palette is then one vector per token).
struct QuadInfo {
    uint32_t tbw, live, fmt;
    int within;
    bool vec;
};
template <int HEAD_DIM>
__device__ __forceinline__ QuadInfo tile_quad_info(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side, int gi, int d0)
{
    QuadInfo q;
    const uint32_t desc = tg.desc[gi];
    q.live = grp_mask(desc);
    q.within = grp_within(desc);
    q.tbw = *(const uint32_t*)&s_tbl[(gi * 2 + side) * HEAD_DIM + d0];
    const uint32_t b0 = q.tbw & 0xffu;
    q.vec = (q.tbw - 0x03020100u) == b0 * 0x01010101u && (b0 & 3u) == 0u;
    q.fmt = ext.fmt[gi][side][(b0 >> 6) & (N_PALETTE - 1)];
    return q;
}

/// Two vector-readable quads as loaded, before conversion: per quad, per
/// token, the words of its four dims' vector (four for F32, two for the
/// 16-bit formats, one for FP8; the rest are not written). `meta` packs
/// what the convert needs — byte 0 and 1 the quads' formats, nibbles 4
/// and 5 their live masks — and `inv` their palette scale reciprocals.
///
/// `W` is the widest format the holder admits, and is what the words cost
/// in registers: a pair converted in the same breath as it is loaded
/// (`tile_quads_vec`, the K side) holds four, so F32 rides the vector
/// path too; a pair held in flight across other work (the V side, loaded
/// ahead of QK and converted after the softmax) holds two — the width of
/// the formats the arenas actually store, `i8_is_narrow_dtype_format` —
/// so the held words cost half the converted floats' registers rather
/// than all of them. F32 on that side decodes through the block path.
template <int W>
struct QuadRaw {
    uint32_t w[2][GROUP_TOK][W];
    uint32_t meta;
    float inv[2];
};

__device__ __forceinline__ uint32_t quad_raw_meta(
    uint32_t fmt0, uint32_t fmt1, uint32_t live0, uint32_t live1)
{
    return fmt0 | (fmt1 << 8) | (live0 << 16) | (live1 << 20);
}

/// Where a pair of vector-readable quads is read from: per quad its
/// palette base, rank, first token and format. A DEAD quad (live mask 0)
/// takes the live quad's address and format so its loads issue with the
/// other's instead of behind a branch of their own — the convert discards
/// its words. `any` is false when neither quad is live.
struct QuadPair {
    const char* base[2];
    int rank[2];
    int within[2];
    uint32_t fmt[2];
    bool any;
};

template <int HEAD_DIM, int W>
__device__ __forceinline__ QuadPair tile_quads_vec_resolve(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2], QuadRaw<W>& raw)
{
    QuadPair pr;
    const bool d0 = q[0].live == 0u, d1 = q[1].live == 0u;
    pr.any = !(d0 && d1);
    // Selects, not an index into q[] — a runtime index would put the
    // pair in local memory.
    const uint32_t tbw0 = d0 ? q[1].tbw : q[0].tbw;
    const uint32_t tbw1 = d1 ? q[0].tbw : q[1].tbw;
    const int g0 = d0 ? gi[1] : gi[0];
    const int g1 = d1 ? gi[0] : gi[1];
    pr.within[0] = d0 ? q[1].within : q[0].within;
    pr.within[1] = d1 ? q[0].within : q[1].within;
    pr.fmt[0] = d0 ? q[1].fmt : q[0].fmt;
    pr.fmt[1] = d1 ? q[0].fmt : q[1].fmt;
    const int p0 = (int)(tbw0 >> 6) & (N_PALETTE - 1);
    const int p1 = (int)(tbw1 >> 6) & (N_PALETTE - 1);
    pr.base[0] = ext.gbase[g0][side][p0];
    pr.base[1] = ext.gbase[g1][side][p1];
    pr.rank[0] = (int)(tbw0 & 63u);
    pr.rank[1] = (int)(tbw1 & 63u);
    raw.inv[0] = 1.f / ext.scl[g0][side][p0];
    raw.inv[1] = 1.f / ext.scl[g1][side][p1];
    raw.meta = quad_raw_meta(pr.fmt[0], pr.fmt[1], q[0].live, q[1].live);
    return pr;
}

/// The loads of quad `i`: one aligned vector per token, four back to
/// back. The group's four tokens lie inside the slice (an aligned quad
/// of it), so a masked token's load issues with the others.
template <int HEAD_DIM, typename Tag, int W>
__device__ __forceinline__ void quad_raw_load(
    Tag tag, const QuadPair& pr, int i, uint32_t (&w)[GROUP_TOK][W])
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    const auto quad = i8_tag_quad(tag, pr.base[i], pr.rank[i], SUB, 1.f);
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) quad.raw4(pr.within[i] + j, w[j]);
}

/// The convert of one quad: xi[k][j] is dim k at token j. A dead quad is
/// zero. A masked token of a live quad is zeroed on the V side (`side`
/// 1): the tile's per-dim scale is an absmax over all its tokens, and the
/// bytes of a token the slice never wrote are whatever the arena held. On
/// the K side it is kept as read: K's scale is per token, so a masked
/// token's bytes reach only its own column, whose score the softmax
/// forces to -inf regardless of what the column holds — even a NaN, which
/// the select discards. (The K callers skip a dead group warp-uniformly
/// before loading, so a dead quad never reaches this on side 0.)
template <typename Tag, int W>
__device__ __forceinline__ void quad_raw_cvt(
    Tag, const uint32_t (&w)[GROUP_TOK][W], float inv, uint32_t live, int side,
    float (&xi)[4][GROUP_TOK])
{
    using E = typename Tag::Elem;
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) {
        // A dead token is masked on its packed words, before the conversion
        // (one select per word, not per value): the zero word of every dtype
        // converts to +0.f, the value the mask used to select.
        const bool l = side == 0 || ((live >> j) & 1u);
        uint32_t wm[W];
        #pragma unroll
        for (int k = 0; k < W; ++k) wm[k] = l ? w[j][k] : 0u;
        float o[4];
        i8_dtype_cvt4<E>(wm, o);
        #pragma unroll
        for (int k = 0; k < 4; ++k) xi[k][j] = o[k];
    }
    if (inv != 1.f) {
        #pragma unroll
        for (int k = 0; k < 4; ++k)
            #pragma unroll
            for (int j = 0; j < GROUP_TOK; ++j) xi[k][j] *= inv;
    }
}

/// The vector path's format set, by the width of words a holder keeps per
/// token: `QuadRaw<4>` admits every dtype format, `QuadRaw<2>` the narrow
/// ones.
template <int W>
__device__ __forceinline__ bool quad_raw_admits(uint32_t fmt)
{
    static_assert(W == 2 || W == 4, "a QuadRaw holds two or four words per token");
    if constexpr (W == 4) return i8_is_dtype_format(fmt);
    else                  return i8_is_narrow_dtype_format(fmt);
}
template <int W, typename F>
__device__ __forceinline__ void quad_raw_dispatch(uint32_t fmt, F&& f)
{
    if constexpr (W == 4) i8_with_dtype_format((int)fmt, f);
    else                  i8_with_narrow_dtype_format((int)fmt, f);
}

/// Run `f(tag, lo, hi)` for the quads lo..hi of the pair under their
/// format's tag: one dispatch for both when their formats agree, which
/// they do whenever both groups sit in one arena block, so the body sees
/// both quads at once and can put every load ahead of every convert.
template <int W, typename F>
__device__ __forceinline__ void quad_pair_dispatch(uint32_t fmt0, uint32_t fmt1, F&& f)
{
    if (fmt0 == fmt1) {
        quad_raw_dispatch<W>(fmt0, [&](auto tag) { f(tag, 0, 1); });
    } else {
        quad_raw_dispatch<W>(fmt0, [&](auto tag) { f(tag, 0, 0); });
        quad_raw_dispatch<W>(fmt1, [&](auto tag) { f(tag, 1, 1); });
    }
}

/// Issue the loads of two vector-readable dtype quads — all eight back
/// to back — into `raw`, to be converted later by `tile_quads_vec_cvt`.
/// A pair with no live quad loads nothing.
///
/// Every word is written before the loads issue. The format arms are
/// if-converted into predicated loads, and a predicated load only
/// partially defines its register, so without a full definition ahead of
/// them the words' live ranges reach back through the tile loop to the
/// previous window's — pinning the pair's registers across the whole tile,
/// K side included, rather than from here to the convert.
template <int HEAD_DIM, int W>
__device__ __forceinline__ void tile_quads_vec_load(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2], QuadRaw<W>& raw)
{
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < GROUP_TOK; ++j)
            #pragma unroll
            for (int k = 0; k < W; ++k) raw.w[i][j][k] = 0u;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, side, gi, raw);
    if (!pr.any) return;
    quad_pair_dispatch<W>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_load<HEAD_DIM>(tag, pr, i, raw.w[i]);
    });
}

/// Convert a loaded pair: x[i][k][j] is quad i's dim k at token j.
template <int W>
__device__ __forceinline__ void tile_quads_vec_cvt(
    const QuadRaw<W>& raw, int side, float (&x)[2][4][GROUP_TOK])
{
    const uint32_t live[2] = { (raw.meta >> 16) & 0xfu, (raw.meta >> 20) & 0xfu };
    if ((live[0] | live[1]) == 0u) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) x[i][k][j] = 0.f;
        return;
    }
    quad_pair_dispatch<W>(raw.meta & 0xffu, (raw.meta >> 8) & 0xffu, [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], side, x[i]);
    });
}

/// One token of one quad on the vector path: o[k] is dim d0 + k of group
/// gi at its token j, converted and scaled; zero for a dead quad or a
/// masked token. The V warps read the ninth quad's tokens through this, a
/// token per lane-octet, under the vote that put the tile on the vector
/// path (every live quad vector-readable, the ninth included).
template <int HEAD_DIM, int W>
__device__ __forceinline__ void tile_quad_vec_token(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side, int gi, int d0,
    int j, float (&o)[4])
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int k = 0; k < 4; ++k) o[k] = 0.f;
    const QuadInfo q = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, side, gi, d0);
    if (((q.live >> j) & 1u) == 0u) return;
    const int p = (int)(q.tbw >> 6) & (N_PALETTE - 1);
    const int rank = (int)(q.tbw & 63u);
    const float inv = 1.f / ext.scl[gi][side][p];
    uint32_t w[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) w[k] = 0u;
    quad_raw_dispatch<W>(q.fmt, [&](auto tag) {
        using E = typename decltype(tag)::Elem;
        const auto quad = i8_tag_quad(tag, ext.gbase[gi][side][p], rank, SUB, 1.f);
        quad.raw4(q.within + j, w);
        i8_dtype_cvt4<E>(w, o);
    });
    if (inv != 1.f) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) o[k] *= inv;
    }
}

/// Two quads of one side: x[i][k][j] is dim d0[i] + k of group gi[i] at
/// its token j. The quad's four rank bytes are one word of the rank table
/// (d0 is a multiple of 4). A dead group's quad is zero; a masked token of
/// a live group reads as 0.
///
/// `tile_quads_probe<HEAD_DIM, W>` reads both quads' descriptors and says
/// whether they take the vector path: every live quad a vector-readable
/// quad of a format a `QuadRaw<W>` admits, loaded by `tile_quads_vec_load`
/// and converted by `tile_quads_vec_cvt` in a few dozen instructions —
/// `tile_quads_vec` is the two back to back. A warp whose lanes do not all
/// take it decodes its whole group by the block path below instead.
template <int HEAD_DIM, int W>
__device__ __forceinline__ bool tile_quads_probe(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, int side,
    const int (&gi)[2], const int (&d0)[2], QuadInfo (&q)[2])
{
    bool fast = true;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        q[i] = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, side, gi[i], d0[i]);
        fast = fast && (q[i].live == 0u || (q[i].vec && quad_raw_admits<W>(q[i].fmt)));
    }
    return fast;
}

template <int HEAD_DIM>
__device__ __forceinline__ void tile_quads_vec(
    const TileExt& ext, const QuadInfo (&q)[2], int side, const int (&gi)[2],
    float (&x)[2][4][GROUP_TOK])
{
    // Load and convert under ONE dispatch: the words then live only inside
    // the format's own arm, at that format's width, rather than at F32's
    // four per token across a dispatch boundary.
    QuadRaw<4> raw;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, side, gi, raw);
    if (!pr.any) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            #pragma unroll
            for (int k = 0; k < 4; ++k)
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) x[i][k][j] = 0.f;
        return;
    }
    const uint32_t live[2] = { q[0].live, q[1].live };
    quad_pair_dispatch<4>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_load<HEAD_DIM>(tag, pr, i, raw.w[i]);
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], side, x[i]);
    });
}

// ============================================================================
// The block path: decode by (palette, rank), not by dim.
//
// A quantised or channel-mapped group is decoded SLOT-wise — a slot is one
// (palette, rank), the unit the arena actually stores (a quant palette's
// block, a dtype palette's column) — and every slot of a palette shares
// the palette's format, so a warp that takes a palette's 64 slots two per
// lane decodes them under ONE warp-uniform format dispatch: no lane ever
// waits for another lane's format body. The slot's dim comes from the
// group's INVERSE rank table (dim of (palette, rank)); the seal gives
// every palette exactly SUB dims, so every slot of a live group is a live
// dim and the inverse is a permutation.
//
//   K — warp = group, a lane the ranks lane and lane + 32 of all four
//       palettes; the decoded tokens are staged as half2 pairs by dim,
//       read back by quad (the RoPE pairing), then rotated and quantised
//       exactly as the vector path's quads are.
//   V — warp = palette, a lane the ranks lane and lane + 32 across all
//       eight groups, held as half2 pairs; the per-dim tile absmax is
//       gathered through a shared atomic max because a slot's dim is the
//       GROUP's map's (a sparse tile's groups can come from slices sealed
//       under different maps), and the quantised words go straight into
//       the transposed slab.
//
// The format bodies are the bulk of the kernel's code — one copy of K's
// and one of V's, each instantiated for every arena format — and are in
// line: a call's arguments and results go through local memory, and a
// sealed tile pays that on every group.
// ============================================================================

/// One arena element as a read-only global load (`__ldg`) — see the
/// I8BlockQuad note in int8_elem.cuh — of whichever width the element is.
template <typename E>
__device__ __forceinline__ E arena_elem(const E* p)
{
    if constexpr (sizeof(E) == 4) {
        const unsigned int w = __ldg(reinterpret_cast<const unsigned int*>(p));
        return *reinterpret_cast<const E*>(&w);
    } else if constexpr (sizeof(E) == 2) {
        const unsigned short w = __ldg(reinterpret_cast<const unsigned short*>(p));
        return *reinterpret_cast<const E*>(&w);
    } else {
        static_assert(sizeof(E) == 1, "arena elements are 1, 2 or 4 bytes");
        const unsigned char w = __ldg(reinterpret_cast<const unsigned char*>(p));
        return *reinterpret_cast<const E*>(&w);
    }
}

/// Tokens within..within+3 of rank `rank` of a palette of format `Tag`,
/// scaled by `inv` (the reciprocal palette scale): a quant palette's
/// block quad, a dtype palette's four strided elements.
template <typename Tag>
__device__ __forceinline__ void pal_rank_load4(
    Tag, const char* base, int rank, int sub, float inv, int within, float (&o)[GROUP_TOK])
{
    if constexpr (I8IsDtypeTag<Tag>::value) {
        using E = typename Tag::Elem;
        const E* p = reinterpret_cast<const E*>(base) + (int64_t)within * sub + rank;
        #pragma unroll
        for (int j = 0; j < GROUP_TOK; ++j) o[j] = to_float<E>(arena_elem(p + j * sub)) * inv;
    } else {
        using B = typename Tag::Block;
        I8BlockQuad<B>::load4(reinterpret_cast<const B*>(base + (int64_t)rank * sizeof(B)), within, inv, o);
    }
}

/// Where dim `d` of a K group is staged: 8 bytes per dim (the group's
/// four tokens as two half2 words), dims below HEAD_DIM / 2 at `lo`, the
/// rest at `hi`.
template <int HEAD_DIM>
__device__ __forceinline__ uint8_t* tile_stage_ptr(uint8_t* lo, uint8_t* hi, int d)
{
    return d < HEAD_DIM / 2 ? lo + d * 8 : hi + (d - HEAD_DIM / 2) * 8;
}

__device__ __forceinline__ uint32_t half2_bits(float a, float b)
{
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

/// K, block path: decode the `NP` palettes from `p0` of group g — all of
/// one format `tag` — and stage each dim's four tokens at
/// `tile_stage_ptr(d)`. Every slot's loads issue before any slot's store
/// (a store sits between the loads otherwise, and each slot then waits a
/// full memory latency for the one before it); `NP` bounds the raw words
/// held across that gap.
template <int HEAD_DIM, int NP, typename Tag>
__device__ __forceinline__ void tile_k_block_decode_palettes(
    Tag tag, const TileExt& ext, const uint8_t* s_inv_g, int g, int within, int lane,
    uint8_t* stg_lo, uint8_t* stg_hi, int stage_half, int p0)
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    uint2 hw[NP][2];
    #pragma unroll
    for (int pi = 0; pi < NP; ++pi) {
        const char* base = ext.gbase[g][0][p0 + pi];
        const float inv = 1.f / ext.scl[g][0][p0 + pi];
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            float o[GROUP_TOK];
            pal_rank_load4(tag, base, lane + WARP_SIZE * s, SUB, inv, within, o);
            hw[pi][s] = make_uint2(half2_bits(o[0], o[1]), half2_bits(o[2], o[3]));
        }
    }
    #pragma unroll
    for (int pi = 0; pi < NP; ++pi)
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int d = s_inv_g[(p0 + pi) * SUB + lane + WARP_SIZE * s];
            if (stage_half < 0) {
                *reinterpret_cast<uint2*>(tile_stage_ptr<HEAD_DIM>(stg_lo, stg_hi, d)) = hw[pi][s];
            } else if ((d >= HEAD_DIM / 2) == (stage_half == 1)) {
                // One half's dims only, all into the private buffer.
                *reinterpret_cast<uint2*>(stg_lo + (d - stage_half * (HEAD_DIM / 2)) * 8) = hw[pi][s];
            }
        }
}

/// K, block path, warp = group g: decode every slot of the group's K and
/// stage each dim's four tokens at `tile_stage_ptr(d)`. `s_inv_g` is the
/// group's K inverse table; `within` the group's first token (an aligned
/// quad of the slice). Masked tokens are staged as read — K's scale is
/// per token, and a masked column's score is forced to -inf.
///
/// A group whose four palettes share a format — nearly every group —
/// decodes them two at a time in one format body (four slots' raw words
/// fit the registers where eight would spill; the fence stops the
/// compiler hoisting the second pair's loads over the first pair's
/// stores). Otherwise each palette runs its own format's body alone: a
/// body per format over all four palettes would decode the others'
/// slots again for nothing, and a palette's loads cannot be hoisted
/// above the previous palette's format dispatch either way.
///
/// `stage_half` −1 stages every dim, the lower half's at `stg_lo` and the
/// upper's at `stg_hi` (`tile_stage_ptr`); 0 or 1 stages only that half's
/// dims, all of them into `stg_lo`, for a group whose own K rows cannot serve
/// as the upper half's stage (see the caller).
template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_block_decode(
    const TileExt& ext, const uint8_t* s_inv_g, int g, int within, int lane,
    uint8_t* stg_lo, uint8_t* stg_hi, int stage_half)
{
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    static_assert(SUB == 2 * WARP_SIZE, "a lane holds two ranks of each palette");
    static_assert(N_PALETTE % 2 == 0, "palettes decode in pairs");
    if ((within & 3) != 0) __trap();
    const uint32_t f4 = *reinterpret_cast<const uint32_t*>(&ext.fmt[g][0][0]);
    const uint32_t f = f4 & 0xffu;
    if (f4 == f * 0x01010101u) {
        i8_with_format((int)f, [&](auto tag) {
            #pragma unroll
            for (int pp = 0; pp < N_PALETTE; pp += 2) {
                if (pp > 0) asm volatile("" ::: "memory");
                tile_k_block_decode_palettes<HEAD_DIM, 2>(tag, ext, s_inv_g, g, within, lane,
                                                          stg_lo, stg_hi, stage_half, pp);
            }
        });
    } else {
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p)
            i8_with_format((int)((f4 >> (8 * p)) & 0xffu), [&](auto tag) {
                tile_k_block_decode_palettes<HEAD_DIM, 1>(tag, ext, s_inv_g, g, within, lane,
                                                          stg_lo, stg_hi, stage_half, p);
            });
    }
}

/// Read a staged quad back: xi[k][j] is dim d0 + k at token j.
/// A quad's staged dims, raw: 4 dims × 4 tokens of half, 8 words. Held
/// this way across a second staging round it is half the registers of the
/// converted floats.
template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_stage_raw(
    uint8_t* stg_lo, uint8_t* stg_hi, int d0, uint4 (&w)[2])
{
    const uint8_t* p = tile_stage_ptr<HEAD_DIM>(stg_lo, stg_hi, d0);
    w[0] = *reinterpret_cast<const uint4*>(p);
    w[1] = *reinterpret_cast<const uint4*>(p + 16);
}

__device__ __forceinline__ void tile_k_stage_cvt(const uint4 (&w)[2], float (&xi)[4][GROUP_TOK])
{
    const uint32_t ww[4][2] = { { w[0].x, w[0].y }, { w[0].z, w[0].w },
                                { w[1].x, w[1].y }, { w[1].z, w[1].w } };
    #pragma unroll
    for (int k = 0; k < 4; ++k)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) {
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&ww[k][h]));
            xi[k][2 * h] = f2.x;
            xi[k][2 * h + 1] = f2.y;
        }
}

template <int HEAD_DIM>
__device__ __forceinline__ void tile_k_stage_read(
    uint8_t* stg_lo, uint8_t* stg_hi, int d0, float (&xi)[4][GROUP_TOK])
{
    uint4 w[2];
    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_hi, d0, w);
    tile_k_stage_cvt(w, xi);
}

__device__ __forceinline__ uint4 sel4(bool keep, const uint4 a, const uint4 b)
{
    return make_uint4(keep ? a.x : b.x, keep ? a.y : b.y, keep ? a.z : b.z, keep ? a.w : b.w);
}

/// V, block path, one slot (palette p, rank `rank`) across the tile's
/// groups: w[g][h] holds tokens 4g + 2h, +1 as a half2 pair — zero for a
/// dead group or a masked token — and each live group's absmax is folded
/// into the tile absmax of the dim its map gives the slot, in `s_vabs`.
/// `NS` is the slot count decoded: TILE_GROUPS on an eight-quad tile (the
/// ninth slot is dead and costs nothing), TILE_SLOTS when the ninth is live.
/// Chosen under a block-uniform branch by the caller, so neither
/// instantiation carries a runtime predicate over its arrays.
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_decode(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_inv, int* s_vabs,
    uint32_t live_g, int p, int rank, uint32_t (&w)[TILE_SLOTS][GROUP_TOK / 2])
{
    static_assert(NS == TILE_GROUPS || NS == TILE_SLOTS, "eight quads, or eight and the ninth");
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int g = 0; g < TILE_SLOTS; ++g)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) w[g][h] = 0u;
    uint32_t rem = live_g;
    while (rem != 0u) {
        const uint32_t f = ext.fmt[__ffs(rem) - 1][1][p];
        uint32_t sel = 0u;
        #pragma unroll
        for (int g = 0; g < NS; ++g)
            if (((rem >> g) & 1u) != 0u && ext.fmt[g][1][p] == f) sel |= 1u << g;
        // Every group decodes in straight-line code — one that is dead or
        // of another format reads the first selected group's slot again
        // and drops the result — so the eight slot loads issue back to
        // back. A skip around a group's loads compiles to a branch, and
        // the next group's loads then wait behind this group's latency.
        const int g0 = __ffs(sel) - 1;
        i8_with_format((int)f, [&](auto tag) {
            // The atomics go out after every group is decoded: a load is
            // not moved across an atomic, so one inside the loop would
            // hold the next group's loads behind this group's latency. A
            // dtype palette's slot is four loads where a block's is two,
            // so those bodies keep half the groups in flight at a time
            // (the fence between the halves is what stops the compiler
            // hoisting the second half's loads and spilling for them).
            constexpr int IN_FLIGHT = I8IsDtypeTag<decltype(tag)>::value ? TILE_GROUPS / 2 : TILE_GROUPS;
            float a[TILE_SLOTS];
            // The ninth slot (NS == TILE_SLOTS) rides in the LAST round, its
            // load in flight with that round's: neither an extra round (a
            // fence and an exposed latency per call) nor a runtime skip
            // around it (which would put the whole word array under a
            // predicate, and so in local memory — measured at the gate as
            // 1.5 KB of spills). `gn` folds at unroll time.
            #pragma unroll
            for (int gb = 0; gb < TILE_GROUPS; gb += IN_FLIGHT) {
                if (gb > 0) asm volatile("" ::: "memory");
                const int gn = (NS > TILE_GROUPS && gb + IN_FLIGHT >= TILE_GROUPS) ? IN_FLIGHT + 1
                                                                                   : IN_FLIGHT;
                #pragma unroll
                for (int gi = 0; gi < IN_FLIGHT + 1; ++gi) {
                    if (gi >= gn) break;
                    const int g = gb + gi;
                    const bool on = ((sel >> g) & 1u) != 0u;
                    const int gs = on ? g : g0;
                    const uint32_t desc = tg.desc[gs];
                    const uint32_t live = grp_mask(desc);
                    float o[GROUP_TOK];
                    pal_rank_load4(tag, ext.gbase[gs][1][p], rank, SUB, 1.f / ext.scl[gs][1][p],
                                   grp_within(desc), o);
                    #pragma unroll
                    for (int j = 0; j < GROUP_TOK; ++j)
                        if (((live >> j) & 1u) == 0u) o[j] = 0.f;
                    a[g] = fmaxf(fmaxf(fabsf(o[0]), fabsf(o[1])), fmaxf(fabsf(o[2]), fabsf(o[3])));
                    #pragma unroll
                    for (int h = 0; h < GROUP_TOK / 2; ++h)
                        w[g][h] = on ? half2_bits(o[2 * h], o[2 * h + 1]) : w[g][h];
                }
            }
            #pragma unroll
            for (int g = 0; g < NS; ++g) {
                const bool on = ((sel >> g) & 1u) != 0u;
                const int gs = on ? g : g0;
                const int d = s_inv[(gs * 2 + 1) * HEAD_DIM + p * SUB + rank];
                // Non-negative floats order as their bit patterns.
                if (on) atomicMax(&s_vabs[d], __float_as_int(a[g]));
            }
        });
        rem &= ~sel;
    }
}

/// V, block path, one slot after the tile absmax is complete: quantise
/// each group's four tokens by their dim's tile scale into that dim's
/// word of the transposed slab. A dead group's word is zero (its map is
/// the identity, so the slot's dim is p · SUB + rank there).
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_quantise(
    const uint32_t (&w)[TILE_SLOTS][GROUP_TOK / 2], const TileGroups& tg, const uint8_t* s_inv,
    const int* s_vabs, int8_t* s_v8t, int p, int rank)
{
    static_assert(NS == TILE_GROUPS || NS == TILE_SLOTS, "eight quads, or eight and the ninth");
    constexpr int SUB = HEAD_DIM / N_PALETTE;
    #pragma unroll
    for (int g = 0; g < NS; ++g) {
        // A dead group has no columns; a live one's tokens go to its own,
        // whole and 4-aligned as one word, otherwise a byte per live token
        // (its word would cover a neighbouring group's columns).
        const uint32_t lv = grp_mask(tg.desc[g]);
        if (lv == 0u) continue;
        const int d = s_inv[(g * 2 + 1) * HEAD_DIM + p * SUB + rank];
        const float a = __int_as_float(s_vabs[d]);
        // The same arithmetic as the vector path's `v_quantise`, to the
        // rounding: a token quantises to the same byte whichever path carries
        // it, which a multi-pass tile — vector-readable chunks taken through
        // this path — depends on to match its single-pass equivalent.
        const float scale = a / 127.f;
        const float inv = (scale > 0.f) ? 1.f / scale : 0.f;
        float v[GROUP_TOK];
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) {
            const float2 f2 = __half22float2(*reinterpret_cast<const __half2*>(&w[g][h]));
            v[2 * h] = f2.x;
            v[2 * h + 1] = f2.y;
        }
        const int c0 = (int)tg.col0[g];
        if (lv == 15u && (c0 & 3) == 0) {
            *reinterpret_cast<uint32_t*>(s_v8t + sw_off<TILE_TOK>(d, c0)) = i8_pack4(v, inv);
        } else {
            #pragma unroll
            for (int j = 0; j < GROUP_TOK; ++j)
                if ((lv >> j) & 1u) s_v8t[sw_off<TILE_TOK>(d, c0 + j)] = i8_quant(v[j], inv);
        }
    }
}

/// V, block path, one V warp's whole tile: palette p = the warp, a lane the
/// ranks `lane` and `lane + 32`. The two slots decode through one copy of
/// the format bodies (the loop is not unrolled), their words kept apart by
/// a select; the tile absmax is complete once every V warp is at the V
/// barrier, which orders the atomics before the reads. `NS` as above,
/// chosen by the caller under a block-uniform branch.
template <int HEAD_DIM, int NS>
__device__ __forceinline__ void tile_v_block_path(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_inv, int* s_vabs,
    uint32_t live_g, int p, int lane, int8_t* s_v8t)
{
    uint32_t w0[TILE_SLOTS][GROUP_TOK / 2], w1[TILE_SLOTS][GROUP_TOK / 2];
    #pragma unroll
    for (int gg = 0; gg < TILE_SLOTS; ++gg)
        #pragma unroll
        for (int h = 0; h < GROUP_TOK / 2; ++h) w0[gg][h] = 0u;
    #pragma unroll 1
    for (int s = 0; s < 2; ++s) {
        uint32_t wt[TILE_SLOTS][GROUP_TOK / 2];
        tile_v_block_decode<HEAD_DIM, NS>(ext, tg, s_inv, s_vabs, live_g, p, lane + WARP_SIZE * s, wt);
        #pragma unroll
        for (int gg = 0; gg < TILE_SLOTS; ++gg)
            #pragma unroll
            for (int h = 0; h < GROUP_TOK / 2; ++h) {
                w0[gg][h] = (s == 0) ? wt[gg][h] : w0[gg][h];
                w1[gg][h] = wt[gg][h];
            }
    }
    asm volatile("bar.sync %0, %1;" :: "n"(TILE_VABS_BARRIER), "n"(TILE_V_WARPS * WARP_SIZE) : "memory");
    tile_v_block_quantise<HEAD_DIM, NS>(w0, tg, s_inv, s_vabs, s_v8t, p, lane);
    tile_v_block_quantise<HEAD_DIM, NS>(w1, tg, s_inv, s_vabs, s_v8t, p, lane + WARP_SIZE);
}

/// One 16-byte global → shared copy through the async proxy, waited for by
/// `cp_async_wait` (decode_helpers.cuh). `.ca`: small, reread rows.
__device__ __forceinline__ void tile_cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(src)
                 : "memory");
}

/// `tile_quads_vec` with quad A (dims d0..d0+3, in the head's lower half)
/// read from the group's staged rows — `k_raw` is [token][lower-half dim]
/// at the palette element width — and quad B loaded from the arena as
/// usual, issued first so it is in flight while A is read back.
template <int HEAD_DIM>
__device__ __forceinline__ void tile_quads_vec_staged(
    const TileExt& ext, const QuadInfo (&q)[2], const int (&gi)[2],
    float (&x)[2][4][GROUP_TOK], const uint8_t* k_raw, int d0)
{
    QuadRaw<4> raw;
    const QuadPair pr = tile_quads_vec_resolve<HEAD_DIM>(ext, q, 0, gi, raw);
    const uint32_t live[2] = { q[0].live, q[1].live };
    quad_pair_dispatch<4>(pr.fmt[0], pr.fmt[1], [&](auto tag, int lo, int hi) {
        if (1 >= lo && 1 <= hi) quad_raw_load<HEAD_DIM>(tag, pr, 1, raw.w[1]);
        if (0 >= lo && 0 <= hi) {
            if constexpr (I8IsDtypeTag<decltype(tag)>::value) {
                using E = typename decltype(tag)::Elem;
                constexpr int ROW = (HEAD_DIM / 2) * (int)sizeof(E);
                #pragma unroll
                for (int j = 0; j < GROUP_TOK; ++j) {
                    const uint8_t* p = k_raw + j * ROW + d0 * (int)sizeof(E);
                    if constexpr (sizeof(E) == 2) {
                        const uint2 v = *reinterpret_cast<const uint2*>(p);
                        raw.w[0][j][0] = v.x;
                        raw.w[0][j][1] = v.y;
                    } else if constexpr (sizeof(E) == 1) {
                        raw.w[0][j][0] = *reinterpret_cast<const uint32_t*>(p);
                    } else {
                        const uint4 v = *reinterpret_cast<const uint4*>(p);
                        raw.w[0][j][0] = v.x; raw.w[0][j][1] = v.y;
                        raw.w[0][j][2] = v.z; raw.w[0][j][3] = v.w;
                    }
                }
            } else {
                __trap();   // a staged tile's quads are dtype quads by the vote that staged it
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i)
            if (i >= lo && i <= hi) quad_raw_cvt(tag, raw.w[i], raw.inv[i], live[i], 0, x[i]);
    });
}

// ============================================================================
// K decode
// ============================================================================
// One group's K rows into the tile's K slab: warp = group `kg`, lane = quad.
// A lane's two quads are its four RoPE pairs — half-split: dims 4q..4q+3 with
// 128+4q..; interleaved: dims 8q..8q+7 as (8q+2i, 8q+2i+1) — and both
// pairings read the same four frequencies 4q..4q+3. The lane looks up the
// rotation at the group's first live token in the slot's rung table
// (`rope_table.cuh`) and the per-position step, that table's `LO` row 1, and
// the later tokens' angles follow by rotating the pair in-register — a
// token's angle is never further than three steps from a lookup, so the drift
// is a few ulp, far under the int8 quantisation the row is about to take. A
// pass-through frequency's rotation and step are exactly (1, 0). On the vector
// path the rope values load ahead of the quad loads so every load of the
// group is in flight together; on the block path they load after the decode,
// so the 16 rope values are not live across its format bodies.
// Scale windows: half-split, a lane's quad A lies in window q>>3 and quad B
// in 4 + (q>>3), so the window absmax is a reduce over the 8 lanes sharing
// q>>3; interleaved, the eight dims lie in window q>>2, a reduce over 4
// lanes. Each token's quantised quads land as one word each. A dead group
// is skipped outright (warp-uniform); a masked token inside a live group
// reads as 0 — its column's score is forced to -inf by the softmax, so its
// int8 row is never read.
//
// A free function over explicit operands, not a lambda over the kernel's
// scope: a closure of this size is one NVCC outlines into a real call, which
// puts every captured variable in addressable local memory and spills the
// caller's live registers around the call. Force-inlined at its one site it
// is register code, as the kernel body it sits in.
//
// `s_kstg` is the K staging slab: `span` is the calling warp's private 4-row
// span of it (the warp's index — a warp decoding the ninth quad as a second
// round reuses its own span, its first group's rows already stored); `s_k8`
// and `s_k_scale` are the tile's K slab and its per-window scales.
template <int HEAD_DIM, bool ROPE_INTERLEAVED>
__device__ __forceinline__ void tile_k_decode(
    const TileExt& ext, const TileGroups& tg, const uint8_t* s_tbl, const uint8_t* s_inv,
    uint8_t* s_kstg, int span, int8_t* s_k8, __half (*s_k_scale)[HEAD_DIM / 32],
    const RopeView& rope, int kg, int lane, const uint8_t* k_raw)
{
    const uint32_t klive = grp_mask(tg.desc[kg]);
    if (klive == 0u) return;
    // The first live token: a group at the head of a slice whose offset
    // falls inside it has leading masked tokens, and those sit before the
    // table's first row.
    const int j0 = __ffs(klive) - 1;
    const int rope0 = tg.rope0[kg] + j0;
    const int dA0 = ROPE_INTERLEAVED ? 8 * lane : 4 * lane;
    const int dB0 = ROPE_INTERLEAVED ? dA0 + 4 : dA0 + HEAD_DIM / 2;
    float rc[4], rs[4], dc[4], ds[4];
    // A lane's four rotary frequencies are 4·lane + i in both pairings: the
    // rotation at the group's first live token from the slot's rung table,
    // and the unit step (its `LO` row 1) the token loop below walks by.
    auto load_rope = [&]() {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            rope_cs_at(rope, rope0, 4 * lane + i, rc[i], rs[i]);
            rope_cs_step(rope, 4 * lane + i, dc[i], ds[i]);
        }
    };
    const int kgi[2] = { kg, kg };
    const int kd0[2] = { dA0, dB0 };
    float x[2][4][GROUP_TOK];
    QuadInfo kq[2];
    const bool kfast = tile_quads_probe<HEAD_DIM, 4>(ext, tg, s_tbl, 0, kgi, kd0, kq);
    if (__builtin_expect(__all_sync(0xffffffffu, kfast), 1)) {
        load_rope();
        if (k_raw != nullptr) {
            // The warp's own copies: every lane waits for its groups, then
            // the warp converges so each lane sees the others' rows.
            cp_async_wait<0, true>();
            __syncwarp();
            tile_quads_vec_staged<HEAD_DIM>(ext, kq, kgi, x, k_raw, dA0);
        } else {
            tile_quads_vec<HEAD_DIM>(ext, kq, 0, kgi, x);
        }
    } else {
        // A staged tile is a vector tile by the vote that staged it; the
        // block path's staging slab is holding the copies.
        if (k_raw != nullptr) __trap();
        // The block path stages the group's dims — 8 B per dim, two 4-row
        // spans — before reading them back by quad. The staging slab's rows
        // of the group's index are this warp's alone. A group's rows are its
        // COLUMNS of the K slab, which are its own to scribble on only when
        // the group is whole and on the 4-grid. A quad at a hole's edge
        // shares its four rows with a neighbour another warp may be writing,
        // so it stages one half of the head at a time through the private
        // span instead — a second decode of the group, paid by the two quads
        // a hole costs a window.
        uint8_t* stg_lo = s_kstg + span * 4 * HEAD_DIM;
        const int kc0 = (int)tg.col0[kg];
        const uint8_t* s_inv_g = s_inv + (kg * 2) * HEAD_DIM;
        const int within = grp_within(tg.desc[kg]);
        if (klive == 15u && (kc0 & 3) == 0) {
            uint8_t* stg_hi = reinterpret_cast<uint8_t*>(s_k8) + kc0 * HEAD_DIM;
            tile_k_block_decode<HEAD_DIM>(ext, s_inv_g, kg, within, lane, stg_lo, stg_hi, -1);
            __syncwarp();
            load_rope();
            tile_k_stage_read<HEAD_DIM>(stg_lo, stg_hi, dA0, x[0]);
            tile_k_stage_read<HEAD_DIM>(stg_lo, stg_hi, dB0, x[1]);
            __syncwarp();
        } else {
            // Half h's dims land at stg_lo + (d − h·HEAD_DIM/2)·8:
            // `tile_stage_ptr(lo, hi, d)` reads the upper half at
            // `hi + (d − HEAD_DIM/2)·8`, so `hi = lo` resolves both halves
            // to the private span.
            //
            // What a round stages is read back RAW (8 words a quad) and
            // converted after both rounds, and the rope rows load after
            // them too: the second round's decode then carries 8 live words
            // of the first, not 16 floats and 16 rope values, which is the
            // difference between this path fitting the register budget and
            // it parking the PV accumulators in local memory.
            //
            // Half-split, quad A is always in the lower half and quad B in
            // the upper, so each round reads its own quad — the round is
            // the unrolled loop's index, no runtime predicate. Interleaved,
            // a lane's two quads share a half that depends on the lane, so
            // each round reads both and keeps the round that staged their
            // half by select: written unconditionally on every path, because
            // an array written under a runtime predicate is one NVCC moves
            // to local memory for the whole kernel, vector path included.
            uint4 wa[2] = { make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u) };
            uint4 wb[2] = { make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u) };
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                tile_k_block_decode<HEAD_DIM>(ext, s_inv_g, kg, within, lane, stg_lo, stg_lo, h);
                __syncwarp();
                if constexpr (ROPE_INTERLEAVED) {
                    const bool keep = (dA0 >= HEAD_DIM / 2) == (h == 1);
                    uint4 ta[2], tb[2];
                    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dA0, ta);
                    tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dB0, tb);
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        wa[i] = sel4(keep, ta[i], wa[i]);
                        wb[i] = sel4(keep, tb[i], wb[i]);
                    }
                } else {
                    if (h == 0) tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dA0, wa);
                    else        tile_k_stage_raw<HEAD_DIM>(stg_lo, stg_lo, dB0, wb);
                }
                __syncwarp();
            }
            tile_k_stage_cvt(wa, x[0]);
            tile_k_stage_cvt(wb, x[1]);
            load_rope();
        }
    }
    #pragma unroll
    for (int j = 0; j < GROUP_TOK; ++j) {
        // Step the angle past every token from the first live one on,
        // masked or not, so (rc, rs) is token j's.
        if (j > 0 && j > j0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float c = rc[i] * dc[i] - rs[i] * ds[i];
                rs[i] = rs[i] * dc[i] + rc[i] * ds[i];
                rc[i] = c;
            }
        }
        if (!((klive >> j) & 1u)) continue;   // warp-uniform
        // The token's column is its logical position's, not the group's
        // slot: a masked j is one outside the window.
        const int tok = (int)tg.col0[kg] + j;
        // r[0..3] is quad A rotated, r[4..7] quad B.
        float r[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float c = rc[i], s = rs[i];
            float lo, hi;
            if constexpr (ROPE_INTERLEAVED) {
                lo = (i < 2) ? x[0][2 * i][j] : x[1][2 * i - 4][j];
                hi = (i < 2) ? x[0][2 * i + 1][j] : x[1][2 * i - 3][j];
            } else {
                lo = x[0][i][j];
                hi = x[1][i][j];
            }
            const float ra = lo * c - hi * s;
            const float rb = lo * s + hi * c;
            if constexpr (ROPE_INTERLEAVED) { r[2 * i] = ra; r[2 * i + 1] = rb; }
            else { r[i] = ra; r[4 + i] = rb; }
        }
        float aA = fmaxf(fmaxf(fabsf(r[0]), fabsf(r[1])), fmaxf(fabsf(r[2]), fabsf(r[3])));
        float aB = fmaxf(fmaxf(fabsf(r[4]), fabsf(r[5])), fmaxf(fabsf(r[6]), fabsf(r[7])));
        if constexpr (ROPE_INTERLEAVED) {
            aA = fmaxf(aA, aB);
            aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 1));
            aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, 2));
            aB = aA;
        } else {
            #pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                aA = fmaxf(aA, __shfl_xor_sync(0xffffffffu, aA, off));
                aB = fmaxf(aB, __shfl_xor_sync(0xffffffffu, aB, off));
            }
        }
        const float scaleA = aA / 127.f;
        const float scaleB = aB / 127.f;
        const float invA = (scaleA > 0.f) ? 1.f / scaleA : 0.f;
        const float invB = (scaleB > 0.f) ? 1.f / scaleB : 0.f;
        const float qa[4] = { r[0], r[1], r[2], r[3] };
        const float qb[4] = { r[4], r[5], r[6], r[7] };
        const uint32_t wa = i8_pack4(qa, invA);
        const uint32_t wb = i8_pack4(qb, invB);
        if constexpr (ROPE_INTERLEAVED) {
            *(uint2*)(s_k8 + sw_off<HEAD_DIM>(tok, dA0)) = make_uint2(wa, wb);
            if ((lane & 3) == 0) s_k_scale[tok][lane >> 2] = __float2half(scaleA);
        } else {
            *(uint32_t*)(s_k8 + sw_off<HEAD_DIM>(tok, dA0)) = wa;
            *(uint32_t*)(s_k8 + sw_off<HEAD_DIM>(tok, dB0)) = wb;
            if ((lane & 7) == 0) {
                s_k_scale[tok][lane >> 3] = __float2half(scaleA);
                s_k_scale[tok][(HEAD_DIM / 2 / 32) + (lane >> 3)] = __float2half(scaleB);
            }
        }
    }
}

// ============================================================================
// The kernel
// ============================================================================
template <typename Q_T, typename T, int HEAD_DIM, bool ROPE_INTERLEAVED>
__global__ void __launch_bounds__(TILE_THREADS, TILE_MIN_BLOCKS)
int8_decode_tile_kernel(
    const Q_T* __restrict__ q,                 // [slots, n_q_head, HD], unrotated
    const uint8_t* __restrict__ headers_ptr,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,               // [slots, n_kv_head, HD], unrotated
    const T* __restrict__ v_new,
    const RopeRungs rungs,
    float* __restrict__ partial_acc,           // [slot·n_q_head + qh][split][HD]
    float* __restrict__ partial_ml,            // [slot·n_q_head + qh][split][2]
    QsaSel sel
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 tile decode: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;        // QK k-step windows (also dims/lane)
    constexpr int SUB = HEAD_DIM / N_PALETTE;   // palette band width
    constexpr int VEC = HEAD_DIM / WARP_SIZE;   // new-token scatter dims per lane
    constexpr int PV_H = HEAD_DIM / 8 / TILE_WARPS; // output n-slices per warp
    constexpr int V_WPW = (N_WIN + TILE_V_WARPS - 1) / TILE_V_WARPS; // V windows per V warp
    constexpr int PAL_U32 = HEAD_DIM / 16;      // u32 words per palette map (16 dims each)
    static_assert(N_PALETTE == 4, "rank-table byte packs the palette into 2 bits");
    static_assert(SUB >= 16 && SUB <= 64, "rank needs 6 bits");
    static_assert(TILE_SM_WARPS * 8 == TILE_TOK && PV_H >= 1,
                  "the softmax warps split QK by n-slice; 8 warps split PV by dim");
    static_assert(TILE_GROUPS == TILE_WARPS && PAL_U32 <= 16,
                  "the map-word role is warp = group, lane = (side, word)");
    static_assert(TILE_GROUPS * 2 * N_PALETTE <= TILE_THREADS, "one stager per (group, side, palette)");
    static_assert(HEAD_DIM == 8 * WARP_SIZE,
                  "K decode: warp = group, lane = quad — a lane's 8 dims × 32 lanes cover the head");
    static_assert(HEAD_DIM / TILE_WARPS == 32 && TILE_TOK == 4 * 8,
                  "V decode: warp = 32-dim window, lane = (token quarter, quad)");
    constexpr int EPS = INT8_TILE_ENTRIES_PER_SPLIT;
    static_assert(EPS <= WARP_SIZE, "one lane resolves one entry");
    constexpr int MAX_GROUPS = EPS * GROUP_TOK;
    constexpr int N_STAGERS = TILE_GROUPS * 2 * N_PALETTE;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot_idx = (int)blockIdx.x;
    const int kv_head_idx = (int)blockIdx.y;
    const int split_idx = (int)blockIdx.z;
    const int num_splits = (int)gridDim.z;
    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;
    int hpg = n_q_head / n_kv_head;
    if (hpg < 1) hpg = 1;
    if (hpg > TILE_M_ROWS) return;              // the launcher never routes this
    const int first_q_head = kv_head_idx * hpg;

    // ------------------------------------------------------------------
    // Shared memory: one arena holding the per-tile decode→compute handoff
    // (int8 K/V tiles + scales, the softmax's row posts, the per-group
    // palette rank tables) and the int8 Q slab, staged once in the
    // prologue and re-read as MMA fragments every tile so the registers it
    // would pin are free during the decode. The four MMA slabs are
    // unpadded and XOR-swizzled (`sw_off`): K and Q rows are HEAD_DIM
    // bytes, the transposed V and the P rows TILE_TOK bytes.
    // ------------------------------------------------------------------
    constexpr int TBL_BYTES = TILE_SLOTS * 2 * HEAD_DIM;

    constexpr int ALIGN16 = 15;
    constexpr int OFF_K8 = 0;
    constexpr int OFF_KS = (OFF_K8 + TILE_TOK * HEAD_DIM + ALIGN16) & ~ALIGN16;
    constexpr int OFF_V8T = (OFF_KS + TILE_TOK * N_WIN * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VS = (OFF_V8T + HEAD_DIM * TILE_TOK + ALIGN16) & ~ALIGN16;
    constexpr int OFF_P8 = (OFF_VS + HEAD_DIM * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_RMAX = (OFF_P8 + TILE_M_ROWS * TILE_TOK + ALIGN16) & ~ALIGN16;
    constexpr int OFF_LSUM = (OFF_RMAX + TILE_M_ROWS * TILE_SM_WARPS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_ALPHA = (OFF_LSUM + TILE_M_ROWS * TILE_SM_WARPS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_TBL = (OFF_ALPHA + TILE_M_ROWS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_INV = (OFF_TBL + TBL_BYTES + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VABS = (OFF_INV + TBL_BYTES + ALIGN16) & ~ALIGN16;
    constexpr int OFF_KSTG = (OFF_VABS + 2 * HEAD_DIM * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_Q8 = (OFF_KSTG + TILE_TOK * HEAD_DIM + ALIGN16) & ~ALIGN16;
    constexpr int ARENA_BYTES = OFF_Q8 + TILE_M_ROWS * HEAD_DIM;
    static_assert(ARENA_BYTES + 4 * 1024 <= 64 * 1024, "two blocks' arenas must fit the smallest SM");
    // The K block path stages a group's dims (HEAD_DIM × 8 B = two 4-row
    // spans) across its own K rows and the K staging slab rows of the
    // same index; nothing else writes either during the K phase.
    static_assert(TILE_GROUPS * 4 * HEAD_DIM == TILE_TOK * HEAD_DIM, "K staging fits the two slabs");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    __shared__ __half s_q_scale[TILE_M_ROWS][N_WIN];
    __shared__ TileExt s_ext[2];
    __shared__ TileGroups s_tg[2];
    // A chunk's tile-0 map words, staged by warp 7 for every thread; from
    // tile 1 on each thread carries its own word in a register.
    __shared__ uint32_t s_map[TILE_SLOTS][2][16];       // [group slot][side][word]
    __shared__ uint32_t s_grp[MAX_GROUPS];
    __shared__ GroupSrc s_grp_src[MAX_GROUPS];
    __shared__ int s_n_tiles;
    // The window's live columns accumulated across a multi-pass tile's
    // passes (zero at every tile's start; a single-pass tile reads it as 0).
    __shared__ uint32_t s_colmask;
    // Dense: the slice holding each of this split's windows' base positions
    // (0xffff before the sequence starts), built once in the prologue so a
    // tile's group resolves from one shared read and one independent header
    // load rather than a walk of dependent loads. A split wider than the
    // table walks from the cursor instead.
    constexpr int WIN_TABLE = 256;
    __shared__ uint16_t s_win_slice[WIN_TABLE];
    // A window's first slice header, fetched two tiles ahead by one thread
    // with a 16-byte cp.async: it is the dense walk's first load, which
    // every thread otherwise waits on at the top of the tile. Slot (t & 1)
    // is filled at the top of tile t for tile t + 2 and read at the top of
    // tile t + 1, behind tile t's mid barrier — the fill and the reads never
    // share a slot. `slice` is the header's slice, -1 for none.
    __shared__ __align__(16) TokenSliceHdr s_hdr_pre[2];
    __shared__ int s_hdr_pre_slice[2];
    // Whether group g's side map is a band map (the identity within each
    // palette): the V warps take the vector path only when every live
    // group's V map is.
    __shared__ uint8_t s_band[TILE_SLOTS][2];

    int8_t* s_q8 = reinterpret_cast<int8_t*>(s_arena + OFF_Q8);
    int8_t* s_k8 = reinterpret_cast<int8_t*>(s_arena + OFF_K8);
    auto s_k_scale = reinterpret_cast<__half(*)[N_WIN]>(s_arena + OFF_KS);
    int8_t* s_v8t = reinterpret_cast<int8_t*>(s_arena + OFF_V8T);
    auto s_v_scale = reinterpret_cast<__half*>(s_arena + OFF_VS);
    int8_t* s_p8 = reinterpret_cast<int8_t*>(s_arena + OFF_P8);
    auto s_rmax = reinterpret_cast<float(*)[TILE_SM_WARPS]>(s_arena + OFF_RMAX);
    auto s_lsum = reinterpret_cast<float(*)[TILE_SM_WARPS]>(s_arena + OFF_LSUM);
    auto s_alpha = reinterpret_cast<float*>(s_arena + OFF_ALPHA);
    // Rank byte (palette << 6 | rank) of dim d under group g's side map:
    // s_tbl[(g·2 + side)·HEAD_DIM + d].
    uint8_t* s_tbl = s_arena + OFF_TBL;
    // Its inverse — the dim of slot (palette p, rank r):
    // s_inv[(g·2 + side)·HEAD_DIM + p·SUB + r]. A permutation, since the
    // seal gives every palette exactly SUB dims.
    uint8_t* s_inv = s_arena + OFF_INV;
    // Per-dim V absmax of the tile as float bits, an atomicMax target; two
    // buffers alternate by tile so the next tile's zeroing does not race
    // this tile's readers.
    int* s_vabs = reinterpret_cast<int*>(s_arena + OFF_VABS);

    const int g = lane >> 2;
    const int n0 = (lane & 3) * 2;

    // A split with nothing to attend emits (m, l) only.
    auto emit_null = [&]() {
        if (tid < hpg) {
            const int64_t base =
                ((int64_t)slot_idx * n_q_head + first_q_head + tid) * num_splits + split_idx;
            partial_ml[base * 2] = -1e38f;
            partial_ml[base * 2 + 1] = 0.f;
        }
    };

    // Three independent loads open the block — the slot header, the row's
    // selection count, and this split's first chunk of entries — in that
    // order, so all three are in flight together: the header is the
    // longest chain (every slice lookup hangs off it) and nothing below
    // branches until it lands.
    constexpr int SCATTER_WARP = TILE_WARPS - 2;
    constexpr int STAGE_WARP = TILE_WARPS - 1;
    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const int n_slices = (int)slot.n_slices;
    const int write_slice_idx = (int)slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    // The slot's own rung: its table for K and the unit step, its m² for Q.
    const RopeView rope = rope_view(rungs, slot.rope_rung);

    const bool qsa_on = qsa_active(sel) && !qsa_row_dense(sel, slot_idx);
    const int sel_cnt = qsa_on ? (int)sel.cnt[slot_idx] : 0;
    const uint32_t* sel_entries = sel.entries + (int64_t)slot_idx * sel.stride;
    int w_lo = 0, w_hi = 0, e_lo = 0, e_hi = 0;
    uint32_t ent0 = 0u;
    bool has0 = false;
    if (qsa_on) {
        const int per = (sel_cnt + num_splits - 1) / num_splits;
        e_lo = split_idx * per;
        e_hi = min(sel_cnt, e_lo + per);
        has0 = warp == STAGE_WARP && e_lo + lane < min(e_hi, e_lo + EPS);
        if (has0) ent0 = sel_entries[e_lo + lane];
    }

    if (n_slices == 0) { emit_null(); return; }
    if (n_slices >= (1 << 23)) __trap();   // the group descriptor's slice field

    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, write_slice_idx, n_kv_head);
    const int ws_offset = (int)slice_offset(write_slice_ptr);
    const int ws_len = (int)slice_len(write_slice_ptr);
    const int ws_rope = (int)slice_rope(write_slice_ptr);

    // ------------------------------------------------------------------
    // This split's share. Dense: a contiguous slice range. Sparse: a
    // contiguous run of the row's selection entries, walked EPS entries at
    // a time — warp 7 resolves each chunk to groups. The launcher sizes the
    // split count so one chunk is the common case; a row longer than the
    // split cap allows takes several. Both are block-uniform. Split 0 runs
    // the scatter below even with no share (the token must land whichever
    // split reads it) before emitting null.
    // ------------------------------------------------------------------
    // Dense: the split's share is a contiguous range of LOGICAL WINDOWS —
    // 32-position spans of the token sequence, the new token included — not
    // of slices. `kv_len` counts every position the query attends, and a
    // window's tokens are gathered from whichever slices hold them, so the
    // partition (and everything the softmax derives from it) is the same for
    // every chunk layout of the same tokens.
    const int kv_len = ws_rope + ws_len + 1;
    const int n_windows = (kv_len + TILE_TOK - 1) / TILE_TOK;
    if (!qsa_on) {
        const int per = (n_windows + num_splits - 1) / num_splits;
        w_lo = split_idx * per;
        w_hi = min(n_windows, w_lo + per);
    }
    const bool has_share = qsa_on ? (e_lo < e_hi) : (w_lo < w_hi);

    // ------------------------------------------------------------------
    // Fused new-token scatter (warp 6): the step's K/V land in the write
    // slice. Split 0 always writes them — the token persists whichever
    // splits read it this step — and so does a split whose share reads
    // the write slice, so its own read (behind the prologue barrier) sees
    // the token. Dense: the split holding write_slice_idx. Sparse: the
    // entries ascend by block, so the split whose first and last entries
    // bracket the new token's block is the one that can hold it. Every
    // other split skips the scatter: with the split count sized to the
    // card, a short row is mostly null splits, and a redundant copy from
    // each of them is a same-line store storm from every SM that the one
    // real block's reads then queue behind.
    // ------------------------------------------------------------------
    {
        const int within = ws_offset + ws_len;
        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;
        if (warp == SCATTER_WARP && within < CHUNK_SIZE) {
            bool covers = split_idx == 0;
            if (qsa_on) {
                if (e_lo < e_hi) {
                    const uint32_t qb = qsa_block_of(sel, slot_idx, ws_rope + ws_len);
                    covers |= (sel_entries[e_lo] >> QSA_CELL_BITS) <= qb &&
                              qb <= (sel_entries[e_hi - 1] >> QSA_CELL_BITS);
                }
            } else {
                // The window holding the new token — its position is the
                // sequence's last.
                const int wn = (kv_len - 1) / TILE_TOK;
                covers |= w_lo <= wn && wn < w_hi;
            }
            if (covers) {
                // The new rows and the head record do not depend on each
                // other: all of it goes out in one round behind the slice
                // header, the format bytes beside the arena pointers and
                // the Q row beside K (the R16 format stores both), rather
                // than each behind the last one's landing.
                const int64_t src_base =
                    ((int64_t)slot_idx * n_kv_head + kv_head_idx) * (int64_t)HEAD_DIM;
                const int64_t q_base =
                    ((int64_t)slot_idx * n_q_head + first_q_head) * (int64_t)HEAD_DIM;
                // Each lane's VEC contiguous elements are one 16-byte row
                // segment: one vector load per row, unpacked in registers.
                static_assert(VEC * sizeof(T) == sizeof(uint4) && VEC * sizeof(Q_T) == sizeof(uint4),
                              "a lane's scatter segment is one 16-byte load");
                const uint4 k_raw = *reinterpret_cast<const uint4*>(k_new + src_base + lane * VEC);
                const uint4 v_raw = *reinterpret_cast<const uint4*>(v_new + src_base + lane * VEC);
                const uint4 q_raw = *reinterpret_cast<const uint4*>(q + q_base + lane * VEC);
                const T* k_el = reinterpret_cast<const T*>(&k_raw);
                const T* v_el = reinterpret_cast<const T*>(&v_raw);
                const Q_T* q_el = reinterpret_cast<const Q_T*>(&q_raw);
                float k_regs[VEC], v_regs[VEC], q_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    k_regs[j] = to_f32<T>(k_el[j]);
                    v_regs[j] = to_f32<T>(v_el[j]);
                    q_regs[j] = to_f32<Q_T>(q_el[j]);
                }
                const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);
                const int pal = lane / LANES_PER_PAL;
                const int local_lane = lane % LANES_PER_PAL;
                const uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
                const uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
                const int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
                const int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
                if (k_ptr_p != 0) {
                    char* k_arena = (char*)(uintptr_t)k_ptr_p;
                    char* v_arena = (char*)(uintptr_t)v_ptr_p;
                    const int k_esz = ArenaFormat::float_elem_size(k_fmt);
                    const int v_esz = ArenaFormat::float_elem_size(v_fmt);
                    if (k_fmt == ArenaFormat::R16) {
                        write_regs_to_r16<VEC>(k_arena, 0, within, local_lane, k_regs, q_regs);
                    } else if (k_esz > 0) {
                        write_regs_to_arena<VEC>(k_arena, (int64_t)within * SUB, local_lane,
                                                 k_esz, k_fmt, k_regs);
                    }
                    if (v_esz > 0) {
                        write_regs_to_arena<VEC>(v_arena, (int64_t)within * SUB, local_lane,
                                                 v_esz, v_fmt, v_regs);
                    }
                }
            }
        }
    }

    if (!has_share) { emit_null(); return; }
    int n_tiles = qsa_on ? 0 : (w_hi - w_lo);   // sparse: set from the resolve

    // ------------------------------------------------------------------
    // Tile staging roles.
    //   map word   — every thread: group = warp, side = lane >> 4,
    //                word = lane & 15 (16 dims of the side's palette map).
    //   descriptor — lanes < 8 of every warp: group = warp,
    //                side = (lane >> 2) & 1, palette = lane & 3.
    // Both roles put a thread on the group of its own warp, so a tile's
    // group is resolved ONCE per thread (`group_info`, the dense window
    // walk or the sparse resolve's table) and feeds both its map word and
    // its descriptor. The resolved group carries the slice's KvHead pointer
    // and rope base, so a stager reaches the head record in one global
    // round. A tile past the chunk's last is all dead groups.
    // ------------------------------------------------------------------
    const int st_g = warp, st_side = (lane >> 2) & 1, st_p = lane & 3;
    const bool stager = lane < 2 * N_PALETTE;
    static_assert(TILE_WARPS * 2 * N_PALETTE == N_STAGERS, "a warp's first 8 lanes stage its group");
    const int mw_side = lane >> 4, mw_word = lane & 15;
    struct GroupInfo { uint32_t desc; GroupSrc src; int col0; };
    // The dense walk's slice cursor: the slice holding the base of the last
    // window this thread resolved, seated once by the gallop at the split's
    // first window. A tile asks for its own window and the next one's, and a
    // multi-pass tile asks for its own again after the lookahead, so the
    // cursor is a hint the walk corrects in either direction — by the
    // handful of slices two neighbouring windows can differ by.
    int cur_slice = -1;
    // Group `gi` of tile `t`.
    //
    // Dense: tile t is window [wb, wb + 32) of positions, and its groups are
    // the aligned physical quads of the slices holding those positions, in
    // position order — a whole window inside one slice is 8 full quads exactly
    // as before; a window a hole falls in has a partial quad on each side of
    // it. `gi` past the last quad is dead. The walk is register arithmetic
    // except where it steps to the next slice (one header load).
    //
    // Sparse: the resolve's packed groups, 8 per tile, at column 4·gi.
    // `ahead`: the call is the tile loop's own look-ahead (tile t + 1 from
    // the top of tile t), whose first slice header was prefetched into
    // `s_hdr_pre[t & 1]`; any other caller loads the header itself.
    auto group_info = [&](int t, int gi, bool ahead) -> GroupInfo {
        GroupInfo g = {0u, {0u, 0}, 0};
        if (t >= n_tiles) return g;
        if (qsa_on) {
            g.desc = s_grp[t * TILE_GROUPS + gi];
            g.src = s_grp_src[t * TILE_GROUPS + gi];
            g.col0 = gi * GROUP_TOK;
            return g;
        }
        const int wb = (w_lo + t) * TILE_TOK;
        const int we = min(wb + TILE_TOK, kv_len);
        // The slice's rows holding window positions: [row_lo, row_hi).
        auto rows_of = [&](const TokenSliceHdr& hh, int slice, int& row_lo, int& row_hi) {
            const int off = hh.offset();
            const int rb = hh.rope_base();
            const int eff = tile_slice_eff_len(hh, slice, write_slice_idx);
            row_lo = off + max(wb - rb, 0);
            row_hi = off + min(eff, we - rb);
        };
        // The window's first slice: the table's, resolved in the prologue.
        // Past the table, seat on the cursor's slice (the gallop's, at the
        // split's first window) and step to the slice whose tokens reach the
        // window: back while the slice begins past it, then forward past any
        // that ends before it (the cursor's, when the window moved on) or that
        // the window starts past the end of (an empty writer behind a partial
        // chunk shares its base).
        int s;
        TokenSliceHdr h;
        if (t < WIN_TABLE) {
            const int ws = (int)s_win_slice[t];
            if (ws == 0xffff) return g;
            s = ws;
            if (ahead && s == s_hdr_pre_slice[t & 1]) h = s_hdr_pre[t & 1];
            else h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        } else {
            s = cur_slice < 0
                ? tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, wb)
                : cur_slice;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            while (s > 0 && h.rope_base() > wb) {
                --s;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            }
        }
        int row_lo, row_hi;
        rows_of(h, s, row_lo, row_hi);
        while (row_lo >= row_hi) {
            if (++s >= n_slices) return g;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            if (h.rope_base() >= we) return g;
            rows_of(h, s, row_lo, row_hi);
        }
        cur_slice = s;
        // Quad gi: skip whole slices' worth of quads, then index into the
        // slice that holds it — arithmetic per slice crossed, not per quad.
        int q, rem = gi;
        for (;;) {
            const int here = ((row_hi - 1) >> 2) - (row_lo >> 2) + 1;
            if (rem < here) {
                q = (row_lo & ~3) + rem * GROUP_TOK;
                break;
            }
            rem -= here;
            do {
                if (++s >= n_slices) return g;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
                if (h.rope_base() >= we) return g;
                rows_of(h, s, row_lo, row_hi);
            } while (row_lo >= row_hi);
        }
        uint32_t mask = 0u;
        #pragma unroll
        for (int j = 0; j < GROUP_TOK; ++j)
            if (q + j >= row_lo && q + j < row_hi) mask |= 1u << j;
        g.desc = grp_pack(s, q, mask);
        g.src.kvheads = h.kvheads_ptr();
        g.src.rope_base = h.rope_base() - h.offset();
        g.col0 = (g.src.rope_base + q) - wb;
        return g;
    };
    // How many quads window `t` spans — the tile's pass count is this over
    // TILE_GROUPS. Counted in one pass over the window's slices, each
    // contributing the aligned quads its rows in the window touch: one
    // header load per slice, arithmetic otherwise. (Asking `group_info` for
    // groups until a dead one is the same answer at nine walks' cost, every
    // one of them a chain of dependent loads on a single thread — measured as
    // a per-launch tax that doubled a one-window decode.)
    auto window_groups = [&](int t, bool ahead) -> int {
        if (qsa_on || t >= n_tiles) return 0;
        const int wb = (w_lo + t) * TILE_TOK;
        const int we = min(wb + TILE_TOK, kv_len);
        int s;
        TokenSliceHdr h;
        if (t < WIN_TABLE) {
            const int ws = (int)s_win_slice[t];
            if (ws == 0xffff) return 0;
            s = ws;
            if (ahead && s == s_hdr_pre_slice[t & 1]) h = s_hdr_pre[t & 1];
            else h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        } else {
            s = cur_slice < 0
                ? tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, wb)
                : cur_slice;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            while (s > 0 && h.rope_base() > wb) {
                --s;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            }
        }
        int n = 0;
        for (;;) {
            const int off = h.offset();
            const int rb = h.rope_base();
            const int eff = tile_slice_eff_len(h, s, write_slice_idx);
            const int row_lo = off + max(wb - rb, 0);
            const int row_hi = off + min(eff, we - rb);
            if (row_lo < row_hi) n += ((row_hi - 1) >> 2) - (row_lo >> 2) + 1;
            // A slice reaching the window's end holds the rest of it: no
            // probe of the next slice's header (a dependent load on warp 0's
            // critical path, every tile, for a packed layout).
            if (rb + eff >= we) break;
            if (++s >= n_slices) break;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            if (h.rope_base() >= we) break;
        }
        return n;
    };
    // One map word of a resolved group: its `side` map, 16-dim word `w`
    // (0 for a dead group or a word past the map).
    auto map_word_of = [&](const GroupInfo& g, int side, int w) -> uint32_t {
        if (grp_mask(g.desc) == 0u || w >= PAL_U32) return 0u;
        const uint8_t* hd = get_head_at<HEAD_DIM>(g.src.kvheads, kv_head_idx);
        const uint8_t* pm = side ? kvhead_v_pal_map<HEAD_DIM>(hd) : kvhead_k_pal_map<HEAD_DIM>(hd);
        return reinterpret_cast<const uint32_t*>(pm)[w];
    };
    auto load_map_word = [&](int t, int gi, int side, int w) -> uint32_t {
        return map_word_of(group_info(t, gi, false), side, w);
    };
    // One descriptor item of a resolved group: (side, palette p).
    auto desc_of = [&](const GroupInfo& g, int side, int p, NextTile& nx) {
        nx.desc = g.desc; nx.rope0 = 0; nx.col0 = 0; nx.ptr = 0; nx.scl = 1.f; nx.fmt = 0;
        if (grp_mask(g.desc) != 0u) {
            const uint8_t* hd = get_head_at<HEAD_DIM>(g.src.kvheads, kv_head_idx);
            nx.ptr = side ? kvhead_v_ptr<HEAD_DIM>(hd, p) : kvhead_k_ptr<HEAD_DIM>(hd, p);
            nx.fmt = side ? kvhead_v_fmt<HEAD_DIM>(hd, p) : kvhead_k_fmt<HEAD_DIM>(hd, p);
            nx.scl = side ? kvhead_v_scale<HEAD_DIM>(hd, p) : kvhead_k_scale<HEAD_DIM>(hd, p);
            nx.rope0 = g.src.rope_base + grp_within(g.desc);
            nx.col0 = g.col0;
        }
    };
    auto load_desc = [&](int t, int gi, int side, int p, NextTile& nx) {
        desc_of(group_info(t, gi, false), side, p, nx);
    };
    auto commit_desc = [&](int buf, int gi, int side, int p, const NextTile& nx) {
        s_ext[buf].gbase[gi][side][p] = (const char*)(uintptr_t)nx.ptr;
        s_ext[buf].scl[gi][side][p] = nx.scl;
        s_ext[buf].fmt[gi][side][p] = (uint8_t)nx.fmt;
        if (side == 0 && p == 0) {
            s_tg[buf].desc[gi] = nx.desc;
            s_tg[buf].rope0[gi] = nx.rope0;
            s_tg[buf].col0[gi] = (int8_t)nx.col0;
            if (gi == 0) s_tg[buf].n_groups = nx.ngroups;
        }
    };
    // The tile's pass count comes from its window's quad count: a sparse
    // tile is the resolve's fixed 8 groups, a dense one is walked.
    auto tile_groups = [&](int t, bool ahead) -> int {
        return qsa_on ? TILE_GROUPS : window_groups(t, ahead);
    };
    // Issue tile t's staging loads into registers under the tile-loop
    // roles. Nothing waits on them here: the descriptors are consumed by
    // `stage_prefetch` and `stage_commit`, the map word by the next tile's
    // table build.
    auto stage_issue = [&](int t, NextTile& nx, bool ahead) {
        const GroupInfo g = group_info(t, warp, ahead);
        nx.map = map_word_of(g, mw_side, mw_word);
        nx.desc = 0u; nx.rope0 = 0; nx.col0 = 0; nx.ngroups = 0; nx.ptr = 0; nx.scl = 1.f; nx.fmt = 0;
        if (stager) desc_of(g, st_side, st_p, nx);
        if (tid == 0) nx.ngroups = (t < n_tiles) ? tile_groups(t, ahead) : 0;
    };
    // L2 prefetch of the staged group's span: a dtype palette's 4 tokens
    // (SUB elements each, contiguous), a quant palette's whole block run
    // (every dim's block holds the 4 tokens' bytes).
    auto prefetch_span = [&](const NextTile& nx) {
        if (nx.ptr != 0 && grp_mask(nx.desc) != 0u) {
            const int es = ArenaFormat::float_elem_size(nx.fmt);
            uint64_t lo, hi;
            if (es == 0) {
                lo = nx.ptr;
                hi = lo + (uint64_t)SUB * ArenaAccessor::get_quant_block_bytes(nx.fmt);
            } else {
                lo = nx.ptr + (uint64_t)grp_within(nx.desc) * SUB * es;
                hi = lo + (uint64_t)GROUP_TOK * SUB * es;
            }
            for (uint64_t a = lo & ~(uint64_t)127; a < hi; a += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
        }
    };
    auto stage_prefetch = [&](const NextTile& nx) {
        if (stager) prefetch_span(nx);
    };
    auto stage_commit = [&](int buf, const NextTile& nx) {
        if (stager) commit_desc(buf, st_g, st_side, st_p, nx);
    };
    // Whether the next tile's K lower half is staged through the slab, decided
    // a tile ahead from the descriptors in registers: a live group whose K is
    // one narrow dtype across the four palettes under a band map — the
    // conditions of the vector path, the slab's reader. The decision is the
    // warp's own (it decodes its own group, and the block path — the slab's
    // other user — scribbles only in its own group's span) and rides in a
    // register to the next tile's table phase, where the copies are issued;
    // it replaces the block-wide vote the decode used to take from shared
    // state after the table barrier. Returns the element size, 0 for none.
    //
    // The copies themselves are NOT issued here. Issuing them at this point —
    // a tile ahead, with the V decode, softmax and PV to land under — was
    // measured: fewer instructions, but the stager region's stalls doubled and
    // the tile loop lost 6–17 % at 32K and above, so they issue where they
    // always did.
    auto stage_k_ahead = [&](const NextTile& nx) -> int {
        if constexpr (ROPE_INTERLEAVED) return 0;
        static_assert(N_PALETTE == 4, "stager lanes 0..3 hold side 0's four palettes");
        constexpr int WORDS_PER_BAND = SUB / 16;
        const uint32_t band = 0x55555555u * (uint32_t)(mw_word / WORDS_PER_BAND);
        const bool band0 =
            __all_sync(0xffffffffu, mw_side != 0 || nx.map == 0u || nx.map == band);
        const uint32_t desc = __shfl_sync(0xffffffffu, nx.desc, 0);
        const int f0 = __shfl_sync(0xffffffffu, nx.fmt, 0);
        const bool fmt_ok = __all_sync(0xffffffffu, lane >= N_PALETTE || nx.fmt == f0);
        if (!band0 || !fmt_ok || grp_mask(desc) == 0u
            || !i8_is_narrow_dtype_format((uint32_t)f0))
            return 0;
        return ArenaFormat::float_elem_size(f0);
    };
    // The ninth quad of tile `t` — slot TILE_GROUPS — staged by warp 0 alone
    // into descriptor buffer `buf`, when the window spans exactly TILE_SLOTS
    // quads (`ngroups`, block-uniform, from the tile's staging): its eight
    // descriptor items on the stager lanes, its 32 map words one per lane
    // into s_map for the table build at the tile's top. Otherwise the slot
    // is committed dead. The loads are exposed on warp 0, a softmax warp,
    // between its K decode and the K barrier it waits at anyway.
    auto stage_slot8 = [&](int t, int buf, int ngroups, bool ahead) {
        if (ngroups == TILE_SLOTS) {
            const GroupInfo g = group_info(t, TILE_GROUPS, ahead);
            if (stager) {
                NextTile w;
                desc_of(g, st_side, st_p, w);
                w.ngroups = ngroups;
                commit_desc(buf, TILE_GROUPS, st_side, st_p, w);
            }
            s_map[TILE_GROUPS][mw_side][mw_word] = map_word_of(g, mw_side, mw_word);
        } else if (lane == 0) {
            s_tg[buf].desc[TILE_GROUPS] = 0u;
        }
    };
    // A resolved sparse chunk's tile 0, staged by the resolving warp alone
    // between chunks: the 64 descriptor items two per lane, the 256 map
    // words eight per lane, all loads in flight before the first shared
    // store, and the spans prefetched to L2 as soon as the descriptors
    // land. (The split's first tile is staged by the whole block under the
    // tile loop's roles, in the prologue.)
    auto stage_tile0 = [&]() {
        // Every group of tile 0, resolved up front per lane: the ten items a
        // lane stages (two descriptors, eight map words) span all eight
        // groups, and resolving each behind its own item put ten record-load
        // chains in series. Resolved first (ten walks of one window, its
        // header hot in L1 after the first), the ten record loads issue
        // together. The map words' groups are the unrolled index (registers);
        // the descriptors' two are lane-dependent, so they walk directly
        // rather than index the array.
        GroupInfo g0[TILE_GROUPS];
        #pragma unroll
        for (int gi = 0; gi < TILE_GROUPS; ++gi) g0[gi] = group_info(0, gi, false);
        NextTile w[2];
        uint32_t mws[8];
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = lane + 32 * k;
            desc_of(group_info(0, i >> 3, false), (i >> 2) & 1, i & 3, w[k]);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = lane + 32 * k;
            mws[k] = map_word_of(g0[k], (i >> 4) & 1, i & 15);
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) prefetch_span(w[k]);
        // Item 0 — lane 0's first — is the one `commit_desc` takes the pass
        // count from.
        if (lane == 0) w[0].ngroups = tile_groups(0, false);
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = lane + 32 * k;
            commit_desc(0, i >> 3, (i >> 2) & 1, i & 3, w[k]);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = lane + 32 * k;
            s_map[i >> 5][(i >> 4) & 1][i & 15] = mws[k];
        }
        // A sparse tile is the resolve's fixed TILE_GROUPS groups: no ninth.
        if (lane == 0) s_tg[0].desc[TILE_GROUPS] = 0u;
    };
    // Rank table of (group = warp, side, word) from the tile's committed
    // map word. A palette holds exactly SUB dims (a rank is 6 bits), so
    // the four palette counts of a word pack a byte each and one 16-lane
    // prefix scan of the packed word ranks all four palettes at once.
    //
    // Two maps have the identity table — rank byte d for dim d — and the
    // warp votes past the scan for them: the band partition (palette
    // d / SUB, the map of every float chunk and every chunk sealed under
    // a format override; word w reads 0x55555555 × (w / WORDS_PER_BAND))
    // and the all-zero map of a dead group. A word-wise mixture of the
    // two would give palette 0 more than SUB dims, so a warp whose words
    // all match one or the other holds one of the two maps.
    //
    // The inverse table (dim of slot) is built alongside: the identity is
    // its own inverse, and the scan path scatters each dim's index to its
    // rank byte.
    auto build_tables = [&](int g, uint32_t mw) {
        constexpr int WORDS_PER_BAND = SUB / 16;
        static_assert(SUB % 16 == 0, "a palette band is whole map words");
        const uint32_t band = 0x55555555u * (uint32_t)(mw_word / WORDS_PER_BAND);
        const int tbl_off = (g * 2 + mw_side) * HEAD_DIM;
        if (__all_sync(0xffffffffu, mw == 0u || mw == band)) {
            if (mw_word < PAL_U32) {
                const uint32_t w0 = 0x03020100u + 0x10101010u * (uint32_t)mw_word;
                const uint4 ident = make_uint4(w0, w0 + 0x04040404u, w0 + 0x08080808u, w0 + 0x0c0c0c0cu);
                *(uint4*)&s_tbl[tbl_off + 16 * mw_word] = ident;
                *(uint4*)&s_inv[tbl_off + 16 * mw_word] = ident;
                if (mw_word == 0) s_band[g][mw_side] = 1;
            }
            return;
        }
        if (mw_word == 0) s_band[g][mw_side] = 0;
        // Byte p of `cnt` counts the word's dims in palette p; the
        // exclusive prefix over the side's 16 words is the palette's rank
        // base at this word.
        uint32_t cnt = 0u;
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            const uint32_t x = mw ^ (0x55555555u * (uint32_t)p);
            cnt += (uint32_t)__popc(~(x | (x >> 1)) & 0x55555555u) << (8 * p);
        }
        uint32_t incl = cnt;
        #pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            const uint32_t v = __shfl_up_sync(0xffffffffu, incl, d, 16);
            if (mw_word >= d) incl += v;
        }
        const uint32_t excl = incl - cnt;
        if (mw_word < PAL_U32) {
            // Dim i's rank is its palette's base plus the lower dims of the
            // word in the same palette: the palette broadcast to every 2-bit
            // field, XOR-matched against the word under a below-i mask.
            uint32_t words[4] = { 0u, 0u, 0u, 0u };
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t p = (mw >> (2 * i)) & 3u;
                const uint32_t x = mw ^ (0x55555555u * p);
                const uint32_t below = 0x55555555u & ((1u << (2 * i)) - 1u);
                const uint32_t local = (uint32_t)__popc(~(x | (x >> 1)) & below);
                const uint32_t base = __byte_perm(excl, 0u, 0x4440u | p);
                const uint32_t rb = (p << 6) + base + local;
                constexpr uint32_t INSERT[4] = { 0x3214u, 0x3240u, 0x3410u, 0x4210u };
                words[i >> 2] = __byte_perm(words[i >> 2], rb, INSERT[i & 3]);
                s_inv[tbl_off + (int)(rb & 0xffu)] = (uint8_t)(16 * mw_word + i);
            }
            *(uint4*)&s_tbl[tbl_off + 16 * mw_word] =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
    };
    // ------------------------------------------------------------------
    // Prologue, two chains side by side. Warps 0–6: rotate and quantise
    // the Q rows into the Q slab (a warp per row, a lane per dim of
    // each 32-dim window; the window absmax is a warp reduce). Warp 7:
    // resolve the split's first selection chunk and stage tile 0. One
    // barrier publishes the scatter, s_q8/s_q_scale, s_grp/s_n_tiles and
    // tile 0's descriptors together.
    // ------------------------------------------------------------------
    int chunk_lo = e_lo;   // first unresolved sparse entry
    // Both V absmax buffers start clear (atomic accumulators, read before
    // written); each tile clears the other buffer for the tile after it.
    for (int i = tid; i < 2 * HEAD_DIM; i += TILE_THREADS) s_vabs[i] = 0;
    if (warp < STAGE_WARP) {
        const int q_pos = ws_rope + ws_len;
        // The window scale carries the softmax scale and the change of
        // base, so a dequantised score is the log2-domain logit outright.
        const float q_scale_mul = softmax_scale * TILE_LOG2E / 127.f;
        for (int r = warp; r < TILE_M_ROWS; r += STAGE_WARP) {
            // A pad row (r >= hpg) is all zeros with zero scales: written
            // directly, no rotation and no absmax reduce.
            if (r >= hpg) {
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) s_q8[sw_off<HEAD_DIM>(r, lane + 32 * w)] = 0;
                if (lane < N_WIN) s_q_scale[r][lane] = __float2half(0.f);
                continue;
            }
            float x[N_WIN];
            const Q_T* qrow = q + ((int64_t)slot_idx * n_q_head + first_q_head + r) * HEAD_DIM;
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) x[w] = to_f32<Q_T>(qrow[lane + 32 * w]);
            i8_apply_rope<HEAD_DIM, N_WIN>(x, q_pos, lane, ROPE_INTERLEAVED ? 1 : 0, rope.for_q());
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                float a = fabsf(x[w]);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                const float inv = (a > 0.f) ? 127.f / a : 0.f;
                s_q8[sw_off<HEAD_DIM>(r, lane + 32 * w)] = i8_quant(x[w], inv);
                if (lane == 0) s_q_scale[r][w] = __float2half(a * q_scale_mul);
            }
        }
    } else {
        if (qsa_on) {
            tile_resolve_entries<HEAD_DIM>(ent0, has0, sel, slot_idx, slices_ptr,
                                           n_slices, write_slice_idx, n_kv_head, lane,
                                           s_grp, s_grp_src, &s_n_tiles);
            __syncwarp();
            n_tiles = s_n_tiles;
            if (lane == 0) {
                s_hdr_pre_slice[0] = -1;
                s_hdr_pre_slice[1] = -1;
            }
        } else {
            // The window table: a lane per slice, 32 slices a round from the
            // slice holding the split's first position (one gallop, warp-wide
            // on the same addresses), each lane claiming the windows whose
            // base lies in its slice's positions [rb, rb + eff). An empty
            // slice claims none; a window before the sequence keeps 0xffff.
            // The round that reaches the split's last position is the last.
            for (int i = lane; i < WIN_TABLE; i += WARP_SIZE) s_win_slice[i] = 0xffff;
            __syncwarp();
            const int pos_lo = w_lo * TILE_TOK;
            const int pos_hi = min(w_hi * TILE_TOK, kv_len);
            const int s0 = tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, pos_lo);
            for (int sb = s0; sb < n_slices; sb += WARP_SIZE) {
                const int s = sb + lane;
                const bool has = s < n_slices;
                int rb = 0, eff = 0;
                if (has) {
                    const TokenSliceHdr h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
                    rb = h.rope_base();
                    eff = tile_slice_eff_len(h, s, write_slice_idx);
                }
                if (has && eff > 0) {
                    const int wf = max((rb + TILE_TOK - 1) / TILE_TOK, w_lo);
                    const int wl = min((rb + eff - 1) / TILE_TOK, min(w_hi, w_lo + WIN_TABLE) - 1);
                    for (int w = wf; w <= wl; ++w) s_win_slice[w - w_lo] = (uint16_t)s;
                }
                if (__any_sync(0xffffffffu, has && rb + eff >= pos_hi)) break;
            }
            __syncwarp();
            // Tile 1's first header, for tile 0's look-ahead; tile 0's own
            // is staged by the whole block below without one.
            if (lane == 0) {
                s_hdr_pre_slice[0] = -1;
                const int s1 = (n_tiles > 1) ? (int)s_win_slice[1] : 0xffff;
                s_hdr_pre_slice[1] = (s1 != 0xffff) ? s1 : -1;
                if (s1 != 0xffff) s_hdr_pre[1] = load_token_slice<HEAD_DIM>(slices_ptr, s1, n_kv_head);
            }
        }
    }
    if (qsa_on) chunk_lo = min(e_hi, chunk_lo + EPS);
    __syncthreads();
    if (qsa_on) n_tiles = s_n_tiles;
    // Tile 0 is staged under the tile loop's own roles — one descriptor or
    // map word per thread, every chain in flight at once across the block —
    // rather than by one warp staging ten items a lane in series while the
    // other seven wait at the barrier. (Measured at 8K, three tiles a block:
    // that serial stage was a tenth of the kernel's stall samples.)
    int k_stg_esz = 0;   // K lower half staged for the next tile: element size, 0 for none
    {
        NextTile nx0;
        stage_issue(0, nx0, false);
        stage_prefetch(nx0);
        stage_commit(0, nx0);
        s_map[warp][mw_side][mw_word] = nx0.map;
        k_stg_esz = stage_k_ahead(nx0);
        if (warp == 0) stage_slot8(0, 0, __shfl_sync(0xffffffffu, nx0.ngroups, 0), false);
    }
    __syncthreads();

    const int ns = warp & 3;        // QK n-slice (tokens ns*8 .. ns*8+7) on the softmax warps
    NextTile nx;
    uint32_t mw = 0u;               // this thread's map word of the tile being decoded

    // Per-warp output state: every warp accumulates its own PV_H output
    // slices at dim_base and carries the 16-row running sums (rebuilt each
    // tile from the softmax warps' posts); the running max is live only on
    // the softmax warps, which are the ones that read it.
    const int dim_base = warp * (HEAD_DIM / TILE_WARPS);
    float o_acc[PV_H][4];
    #pragma unroll
    for (int s = 0; s < PV_H; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    // ==================================================================
    // Tile loop. The outer loop re-runs it per resolved sparse chunk; a
    // dense split is one pass.
    // ==================================================================
    int tile_no = 0;   // tiles done across chunks; selects the V absmax buffer
    for (;;) {
    for (int t = 0; t < n_tiles; ++t, ++tile_no) {
        const int cur = t & 1, nxt = cur ^ 1;
        const TileGroups& tg = s_tg[cur];
        const TileExt& ext = s_ext[cur];
        // A window of more quads than one pass holds — a hole off the 4-grid
        // splits a quad in two — is taken in passes that share this tile's
        // softmax and V scale (the pre-sweep below). Block-uniform: the count
        // was committed before the previous tile's last barrier.
        const int n_pass = (tg.n_groups <= TILE_SLOTS)
                               ? 1
                               : (tg.n_groups + TILE_GROUPS - 1) / TILE_GROUPS;

        // -------------------- STAGE --------------------
        // The next tile's loads go out first; this tile's tables are built
        // from its map word — tile 0's from the prologue's staging, later tiles'
        // the word this thread loaded a tile ahead — only for a group whose
        // word differs from the previous tile's (warp-uniform vote; a
        // chunk's tile 0 always builds).
        {
            const uint32_t mw_cur = (t == 0) ? s_map[warp][mw_side][mw_word] : nx.map;
            stage_issue(t + 1, nx, true);
            if (t == 0 || __any_sync(0xffffffffu, mw_cur != mw)) build_tables(warp, mw_cur);
            mw = mw_cur;
            // The ninth quad's tables, by warp 0 from the words its staging
            // left in s_map, whenever the slot is live (block-uniform: the
            // descriptor was committed before the previous tile's last
            // barrier, or in the prologue).
            if (warp == 0 && grp_mask(tg.desc[TILE_GROUPS]) != 0u)
                build_tables(TILE_GROUPS, s_map[TILE_GROUPS][mw_side][mw_word]);
            if (tid == 0) {
                s_colmask = 0u;
                // Two tiles ahead: the slice header tile t + 2's walk starts
                // from, into the slot tile t + 1's look-ahead reads. Waited
                // for before this tile's mid barrier, which publishes it.
                int ps = -1;
                if (!qsa_on && t + 2 < n_tiles && t + 2 < WIN_TABLE) {
                    const int ws = (int)s_win_slice[t + 2];
                    if (ws != 0xffff) ps = ws;
                }
                s_hdr_pre_slice[t & 1] = ps;
                if (ps >= 0) {
                    tile_cp_async_16(&s_hdr_pre[t & 1],
                                     get_slice<HEAD_DIM>(slices_ptr, ps, n_kv_head));
                    cp_async_commit<true>();
                }
            }
        }
        __syncthreads(); // publishes the rank tables
        // Whether the ninth slot is live this tile (block-uniform: committed
        // before the previous tile's last barrier, or in the prologue). The
        // V side and warp 0's second K round key on it, so an eight-quad
        // tile runs code with no trace of the slot.
        const bool has8 = grp_mask(tg.desc[TILE_GROUPS]) != 0u;

        // K lower-half staging: on a single-pass tile whose group's K is a
        // narrow dtype under a band map (`k_stg_esz`, decided a tile ahead
        // by `stage_k_ahead` from the descriptors in registers), the warp
        // copies its group's four rows of dims [0, HEAD_DIM/2) into its span
        // of the slab now with cp.async, and the decode reads quad A from
        // them once the rope rows and quad B's global load are in flight.
        // Half the K bytes' latency then sits under the tile's staging and
        // table work rather than at the head of the decode.
        uint8_t* k_raw = nullptr;
        if (k_stg_esz > 0 && n_pass == 1) {
            k_raw = s_arena + OFF_KSTG + warp * (GROUP_TOK * HEAD_DIM);
            const int within = grp_within(tg.desc[warp]);
            const int row = (HEAD_DIM / 2) * k_stg_esz;   // a token's lower half, bytes
            const int ch = row / 16;                      // its 16-byte chunks
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int q = lane + WARP_SIZE * r;
                if (q < GROUP_TOK * ch) {
                    const int j = q / ch, cq = q % ch;
                    const int p = cq / (ch / 2);          // palette 0 or 1
                    const int off = (cq % (ch / 2)) * 16;
                    const char* src = ext.gbase[warp][0][p]
                                      + (int64_t)(within + j) * SUB * k_stg_esz + off;
                    tile_cp_async_16(k_raw + j * row + cq * 16, src);
                }
            }
            cp_async_commit<true>();
        }

        // ------------------------------------------------------------------
        // K decode, and the multi-pass window. A single-pass tile (every
        // window without a hole off the 4-grid inside it) is one trip of
        // sweep 1 with its groups already staged: the K decode and nothing
        // else. A multi-pass window (n_pass > 1) does not fit one set of
        // group slots, so its passes are published here one at a time, and
        // the tile's regular body runs the last. What makes the passes ONE
        // tile numerically is that they share the softmax (the K rows of
        // every pass sit in their own columns of the K slab when the body's
        // QK runs, and the column mask accumulates in s_colmask) and the V
        // scale (every pass's per-dim absmax folds into the tile's absmax
        // buffer before any pass's V is quantised, so the body's block path
        // quantises the last pass against the window's maximum and sweep 2
        // quantises the earlier ones against the same). The V side takes
        // the block path throughout — its absmax is an atomic fold into
        // that buffer, which is what lets it span passes.
        //
        // Slow, exposed and serial by design: a pass costs a synchronous
        // stage, and the earlier passes' V rows are decoded twice (absmax,
        // then quantise). A window off the 4-grid — every window after a
        // section sealed mid-quad, until the next boundary — spans nine
        // quads and fits one pass through slot TILE_GROUPS; only a window
        // with two holes inside it (more than TILE_SLOTS quads) comes here.
        // ------------------------------------------------------------------
        int* vabs_mp = s_vabs + (tile_no & 1) * HEAD_DIM;
        // Publish pass p's groups into this tile's descriptor buffer and
        // tables, and fold its columns into the tile's mask.
        auto publish = [&](int p) {
            const GroupInfo g = group_info(t, p * TILE_GROUPS + warp, false);
            if (stager) {
                NextTile w;
                desc_of(g, st_side, st_p, w);
                w.ngroups = tg.n_groups;
                commit_desc(cur, st_g, st_side, st_p, w);
                if (st_side == 0 && st_p == 0)
                    atomicOr(&s_colmask, grp_cols(grp_mask(w.desc), w.col0));
            }
            const uint32_t word = map_word_of(g, mw_side, mw_word);
            __syncthreads();   // the previous pass's readers are done with the tables
            build_tables(warp, word);
            mw = word;
            __syncthreads();
        };
        auto v_live = [&]() -> uint32_t {
            const bool l = lane < TILE_SLOTS && grp_mask(tg.desc[lane]) != 0u;
            return __ballot_sync(0xffffffffu, l);
        };
        // Sweep 1: every pass's K rows and V absmax. The K decode's one site.
        for (int p = 0; p < n_pass; ++p) {
            if (n_pass > 1) publish(p);
            tile_k_decode<HEAD_DIM, ROPE_INTERLEAVED>(ext, tg, s_tbl, s_inv, s_arena + OFF_KSTG,
                                                      warp, s_k8, s_k_scale, rope, warp, lane,
                                                      k_raw);
            // The ninth quad (slot TILE_GROUPS, live only on a single-pass
            // nine-quad window): warp 0's second round, through its own
            // staging span, from the group's spans — a dead slot returns at
            // once.
            if (warp == 0 && n_pass == 1 && has8)
                tile_k_decode<HEAD_DIM, ROPE_INTERLEAVED>(ext, tg, s_tbl, s_inv, s_arena + OFF_KSTG,
                                                          0, s_k8, s_k_scale, rope, TILE_GROUPS,
                                                          lane, nullptr);
            if (n_pass > 1) {
                if (warp >= TILE_SM_WARPS) {
                    const int vw = warp - TILE_SM_WARPS;
                    const uint32_t live_g = v_live();
                    uint32_t wt[TILE_SLOTS][GROUP_TOK / 2];
                    tile_v_block_decode<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs_mp, live_g, vw, lane, wt);
                    tile_v_block_decode<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs_mp, live_g, vw,
                                                  lane + WARP_SIZE, wt);
                }
                __syncthreads();
            }
        }
        if (n_pass > 1) {
            // Sweep 2: the earlier passes' V rows, quantised against the
            // window's absmax. The last pass is the body's: its groups are
            // published and go out as the tile's.
            for (int p = 0; p < n_pass; ++p) {
                publish(p);
                if (p + 1 == n_pass) break;
                if (warp >= TILE_SM_WARPS) {
                    const int vw = warp - TILE_SM_WARPS;
                    const uint32_t live_g = v_live();
                    uint32_t w0[TILE_SLOTS][GROUP_TOK / 2], w1[TILE_SLOTS][GROUP_TOK / 2];
                    tile_v_block_decode<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs_mp, live_g, vw, lane, w0);
                    tile_v_block_decode<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs_mp, live_g, vw,
                                                  lane + WARP_SIZE, w1);
                    tile_v_block_quantise<HEAD_DIM, TILE_GROUPS>(w0, tg, s_inv, vabs_mp, s_v8t, vw, lane);
                    tile_v_block_quantise<HEAD_DIM, TILE_GROUPS>(w1, tg, s_inv, vabs_mp, s_v8t, vw,
                                                    lane + WARP_SIZE);
                }
                __syncthreads();
            }
        }

        // The header prefetch has landed by now; wait so the mid barrier
        // publishes it (the K staging's copies were waited for in the
        // decode, by the warp that reads them).
        if (tid == 0) cp_async_wait<0, true>();
        __syncwarp();

        // The group's K rows are stored: the V warps mark them published
        // without waiting (the softmax warps wait below, after their own
        // staging work, so a slow group's decode overlaps the staging).
        if (warp >= TILE_SM_WARPS)
            asm volatile("bar.arrive %0, %1;" :: "n"(TILE_K_BARRIER), "n"(TILE_THREADS) : "memory");

        // The next tile's descriptors have landed by now: warm L2 for its
        // spans while this tile's V decodes and computes, and hand them to
        // the other descriptor buffer — nothing reads it until the next
        // tile's first barrier publishes it, and every warp is past this
        // tile's, so the staged words stop being live here rather than
        // riding through the V decode and the softmax.
        stage_prefetch(nx);
        stage_commit(nxt, nx);
        // Reconverge before the split. The prefetch and the commit run on
        // the stager lanes alone, in loops of per-lane length, and the
        // other lanes are free to run ahead without them: a warp that
        // arrives here diverged stays diverged through the whole V side,
        // which has no full-warp barrier before the tile's last one, and
        // issues every V instruction once per fragment (measured: 16
        // active lanes per V-side instruction, the phase at twice its
        // instruction count, every shuffle through the collective slow
        // path). The softmax warps reconverge at their K barrier below.
        __syncwarp();
        // Decide the next tile's K staging from its descriptors, in registers.
        k_stg_esz = (t + 1 < n_tiles) ? stage_k_ahead(nx) : 0;
        // The next tile's ninth quad, if it has one, into the buffer this
        // commit just filled (tid 0 holds the next tile's quad count).
        if (warp == 0) stage_slot8(t + 1, nxt, __shfl_sync(0xffffffffu, nx.ngroups, 0), true);

        // The tile's middle splits the block two ways: warps 0..3 run QK
        // and the softmax, warps 4..7 read and quantise V at the same
        // time, and the two halves meet at the tile's last barrier. Neither
        // half holds anything of the other's work, so each runs at its own
        // register footprint, and V's load latency is paid under the MMAs
        // and exp2s of the other half rather than on anyone's critical
        // path.
        //
        // QK: warp ns owns the tile's columns ns·8..+7 — N_WIN m16n8k32
        // MMAs over the whole head, the Q A-fragments and both rows'
        // window scales re-read from the Q slab each window (one ldmatrix
        // per window per operand). The accumulator lands in the lane's own
        // softmax cells: s[row][i] is (row g + 8·row, column ns·8 + n0 +
        // i), so no score leaves the warp and no barrier sits between QK
        // and the softmax.
        //
        // Softmax: a masked column is -inf. The row max crosses the four
        // warps through s_rmax under the softmax warps' own named barrier;
        // each warp then exponentiates only its two columns per row, packs
        // them into the int8 P slab, and posts its partial row sum. Warp 0
        // posts the row's rescale factor for the PV warps. The running max
        // is clamped to -1e30 — below any logit, above -inf — so a masked
        // column's exp2(-inf − m) is exactly 0 and a not-yet-started row's
        // exp2(m_run − m_new) likewise, with no guard on either; the
        // null-split test is then "no row sum was ever added" (a live
        // column contributes ≥ 1).
        if (warp < TILE_SM_WARPS) {
            // Wait for every group's K rows (the V warps arrived above).
            asm volatile("bar.sync %0, %1;" :: "n"(TILE_K_BARRIER), "n"(TILE_THREADS) : "memory");
            float s[2][2] = { { 0.f, 0.f }, { 0.f, 0.f } };
            #pragma unroll
            for (int wa = 0; wa < N_WIN; ++wa) {
                uint32_t q_frag[4], b[2];
                ldmatrix_x4_b16(q_frag, sw_a_frag_addr<HEAD_DIM>(s_q8, 32 * wa, lane));
                ldmatrix_x2_b16(b, sw_b_frag_addr<HEAD_DIM>(s_k8, ns * 8, 32 * wa, lane));
                const float qs0 = __half2float(s_q_scale[g][wa]);
                const float qs1 = __half2float(s_q_scale[g + 8][wa]);
                const float ks0 = __half2float(s_k_scale[ns * 8 + n0][wa]);
                const float ks1 = __half2float(s_k_scale[ns * 8 + n0 + 1][wa]);
                int32_t c_i[4] = { 0, 0, 0, 0 };
                int32_t d_i[4];
                mma_int8_m16n8k32(d_i, q_frag, b, c_i);
                s[0][0] += (float)d_i[0] * qs0 * ks0;
                s[0][1] += (float)d_i[1] * qs0 * ks1;
                s[1][0] += (float)d_i[2] * qs1 * ks0;
                s[1][1] += (float)d_i[3] * qs1 * ks1;
            }
            // The window's live columns. A multi-pass tile's earlier passes
            // left theirs in s_colmask; this pass's groups add their own.
            uint32_t mask = s_colmask;
            #pragma unroll
            for (int gi = 0; gi < TILE_SLOTS; ++gi)
                mask |= grp_cols(grp_mask(tg.desc[gi]), (int)tg.col0[gi]);
            const int c = ns * 8 + n0;
            const bool live0 = (mask >> c) & 1u;
            const bool live1 = (mask >> (c + 1)) & 1u;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const int r = g + row * 8;
                s[row][0] = live0 ? s[row][0] : -INFINITY;
                s[row][1] = live1 ? s[row][1] : -INFINITY;
                float mt = fmaxf(s[row][0], s[row][1]);
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
                if ((lane & 3) == 0) s_rmax[r][ns] = mt;
            }
            asm volatile("bar.sync %0, %1;" :: "n"(TILE_SM_BARRIER), "n"(TILE_SM_WARPS * WARP_SIZE) : "memory");
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const int r = g + row * 8;
                const float4 rm = *(const float4*)&s_rmax[r][0];
                const float m_tile = fmaxf(fmaxf(rm.x, rm.y), fmaxf(rm.z, rm.w));
                const float m_new = fmaxf(fmaxf(m_run[row], m_tile), -1e30f);
                const float alpha = exp2f(m_run[row] - m_new);
                const float p0 = exp2f(s[row][0] - m_new);
                const float p1 = exp2f(s[row][1] - m_new);
                float ls = p0 + p1;
                ls += __shfl_xor_sync(0xffffffffu, ls, 1);
                ls += __shfl_xor_sync(0xffffffffu, ls, 2);
                *(uint16_t*)(s_p8 + sw_off<TILE_TOK>(r, c)) = (uint16_t)i8_pack2(p0, p1, 127.f);
                if ((lane & 3) == 0) {
                    s_lsum[r][ns] = ls;
                    if (warp == 0) s_alpha[r] = alpha;
                }
                m_run[row] = m_new;
            }
        } else {
            // V: warp vw = warp − 4 owns the head's 32-dim windows
            // vw·V_WPW .. +V_WPW−1 (two of the eight at HEAD_DIM 256; a
            // window past N_WIN is a whole warp's no-op). Within a window,
            // lane = (token quarter, quad): the lane holds dims win·32 +
            // 4·(lane&7) .. +3 of groups 2·(lane>>3) and +1 — the same
            // quad across two groups, so a window's eight loads go out
            // under one dispatch.
            //
            // Quantise: the PV MMA takes one scale per dim for all 32
            // columns, and the groups come from different arena blocks
            // with different block scales, so each dim's absmax over the
            // tile is the lane's 8 tokens joined across the four
            // quarter-lanes (xor 8, 16); every lane then packs its quad
            // for each of its two groups as one word of the transposed
            // slab, and the quarter-0 lanes write the four dims' scales as
            // one vector. A dead group's quad reads as zeros and masked
            // tokens inside a live group quantise to 0. Either way P = 0
            // for them. The reduction is warp-wide, so it runs after the
            // per-lane decode paths have reconverged.
            const int vw = warp - TILE_SM_WARPS;
            const int vq = lane >> 3;
            const int vg0 = 2 * vq;
            const int vgi[2] = { vg0, vg0 + 1 };
            const auto win_d0 = [&](int h) { return (vw * V_WPW + h) * 32 + (lane & 7) * 4; };
            // Every live group of the tile at its slot's columns 4·g — the
            // packed layout, and every window without a hole inside it. One
            // vote per tile (lane g judges group g) picks `v_quantise`'s
            // store path below.
            bool lane_slotted = true;
            if (lane < TILE_SLOTS) {
                const uint32_t lm = grp_mask(tg.desc[lane]);
                lane_slotted = lm == 0u || (lm == 15u && (int)tg.col0[lane] == lane * GROUP_TOK);
            }
            const bool slotted = __all_sync(0xffffffffu, lane_slotted);
            __syncwarp();
            // `v8` is this lane-octet's token of the ninth quad (zero when the
            // slot is dead or the token masked, so an eight-quad window's
            // arithmetic is untouched); `tok8` says it is live and stores.
            // `ninth` (a std::bool_constant) selects the nine-slot body: an
            // eight-quad tile's instantiation never touches v8 or tok8.
            const auto v_quantise = [&](auto ninth, const float (&v)[2][4][GROUP_TOK],
                                        const float (&v8)[4], bool tok8, int vd0) {
                constexpr bool NINTH = decltype(ninth)::value;
                float scale[4], inv[4];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float a = 0.f;
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg)
                        #pragma unroll
                        for (int j = 0; j < GROUP_TOK; ++j) a = fmaxf(a, fabsf(v[gg][k][j]));
                    if constexpr (NINTH) a = fmaxf(a, fabsf(v8[k]));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
                    scale[k] = a / 127.f;
                    inv[k] = (scale[k] > 0.f) ? 1.f / scale[k] : 0.f;
                }
                // A group's four tokens go to its own columns. A whole,
                // 4-aligned group is one word per dim row — one swizzled
                // address, the four rows at immediate offsets (vd0 is a
                // multiple of 4, so they share a swizzle key). A group that is
                // partial or off the 4-grid — a quad at either side of a hole,
                // or at the window's edge — stores its live tokens a byte at a
                // time: its word would cover a neighbour's columns. A dead
                // group has no columns and stores nothing.
                //
                // The eight words are quantised ONCE, before any branch —
                // the paths below differ only in where they store. (Written
                // per branch, NVCC materialised the quantisation twice on the
                // taken path.)
                uint32_t w[2][4];
                #pragma unroll
                for (int gg = 0; gg < 2; ++gg)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) w[gg][k] = i8_pack4(v[gg][k], inv[k]);
                // The ninth quad's token: a byte per dim row at its own
                // column (its quad straddles the window's edge, so it is
                // never a slotted word). The same code as the packed bytes.
                if constexpr (NINTH) {
                    if (tok8) {
                        const int c8 = (int)tg.col0[TILE_GROUPS] + vq;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            s_v8t[sw_off<TILE_TOK>(vd0 + k, c8)] = i8_quant(v8[k], inv[k]);
                    }
                }
                // `slotted` (a vote per tile, below) says every live group of
                // the tile sits at its slot's columns 4·g — the packed layout,
                // and every window without a hole inside it. That is the
                // straight-line path: one address per lane and eight words,
                // no branch per group; a dead group's words land in columns
                // the mask retires, as they always have.
                if (slotted) {
                    int8_t* vrow = s_v8t + sw_off<TILE_TOK>(vd0, vg0 * GROUP_TOK);
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg)
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            *(uint32_t*)(vrow + k * TILE_TOK + gg * GROUP_TOK) = w[gg][k];
                } else {
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg) {
                        const int g = vg0 + gg;
                        const uint32_t lv = grp_mask(tg.desc[g]);
                        if (lv == 0u) continue;
                        const int c0 = (int)tg.col0[g];
                        if (lv == 15u && (c0 & 3) == 0) {
                            int8_t* vrow = s_v8t + sw_off<TILE_TOK>(vd0, c0);
                            #pragma unroll
                            for (int k = 0; k < 4; ++k)
                                *(uint32_t*)(vrow + k * TILE_TOK) = w[gg][k];
                        } else {
                            // The live tokens' bytes out of the packed word
                            // (token j is byte j): its word would cover a
                            // neighbour's columns.
                            #pragma unroll
                            for (int j = 0; j < GROUP_TOK; ++j) {
                                if (!((lv >> j) & 1u)) continue;
                                #pragma unroll
                                for (int k = 0; k < 4; ++k)
                                    s_v8t[sw_off<TILE_TOK>(vd0 + k, c0 + j)] =
                                        (int8_t)((w[gg][k] >> (8 * j)) & 0xffu);
                            }
                        }
                    }
                }
                if (vq == 0) {
                    const __half2 s01 = __floats2half2_rn(scale[0], scale[1]);
                    const __half2 s23 = __floats2half2_rn(scale[2], scale[3]);
                    *(uint2*)&s_v_scale[vd0] = make_uint2(
                        *reinterpret_cast<const uint32_t*>(&s01),
                        *reinterpret_cast<const uint32_t*>(&s23));
                }
            };
            // The tile's V absmax buffer; the other one, last read by the
            // previous tile's quantise, is cleared here for the next tile.
            const int vt = tid - TILE_SM_WARPS * WARP_SIZE;
            int* vabs = s_vabs + (tile_no & 1) * HEAD_DIM;
            int* vabs_next = s_vabs + ((tile_no & 1) ^ 1) * HEAD_DIM;
            #pragma unroll
            for (int d = vt; d < HEAD_DIM; d += TILE_V_WARPS * WARP_SIZE) vabs_next[d] = 0;
            // The path is decided for the whole V side at once, from the
            // tile's live groups: every live group's V map a band map and
            // every palette a narrow dtype, and the windows take the
            // vector path — a vote per warp could split the head between
            // the two paths' different partitions of it (windows against
            // slots) and write a dim twice or never. Lane g judges group g
            // and the warp ballots, so the decision is a dozen instructions
            // rather than eight groups' worth per lane.
            static_assert(TILE_GROUPS <= WARP_SIZE, "a lane per group");
            bool lane_live = false, lane_vec = true;
            if (lane < TILE_SLOTS) {
                lane_live = grp_mask(tg.desc[lane]) != 0u;
                const uint32_t f4 = *reinterpret_cast<const uint32_t*>(&ext.fmt[lane][1][0]);
                bool narrow = s_band[lane][1] != 0;
                #pragma unroll
                for (int p = 0; p < N_PALETTE; ++p)
                    narrow = narrow && i8_is_narrow_dtype_format((f4 >> (8 * p)) & 0xffu);
                lane_vec = !lane_live || narrow;
            }
            const uint32_t live_g = __ballot_sync(0xffffffffu, lane_live);
            // A multi-pass tile takes the block path whatever its formats:
            // its V scale is the window's absmax, folded across the passes
            // through the absmax buffer, and only the block path scales
            // through that buffer. (The vote runs first — it is collective.)
            const bool vec_all = __all_sync(0xffffffffu, lane_vec);
            const bool vec = vec_all && n_pass == 1;
            __syncwarp();
            if (vec) {
                // The windows are software-pipelined one deep: window
                // h+1's loads go out once window h's words are converted
                // and before its quantise, so at most one window's words
                // and one window's floats are live at once while the next
                // window's latency runs under the quantise.
                QuadInfo q[V_WPW][2];
                #pragma unroll
                for (int h = 0; h < V_WPW; ++h) {
                    if (vw * V_WPW + h < N_WIN) {
                        #pragma unroll
                        for (int i = 0; i < 2; ++i)
                            q[h][i] = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, 1, vgi[i], win_d0(h));
                    }
                }
                QuadRaw<2> raw;
                if (vw * V_WPW < N_WIN) tile_quads_vec_load<HEAD_DIM>(ext, q[0], 1, vgi, raw);
                // Two instantiations of the window loop under the tile's
                // block-uniform ninth-slot flag: a nine-quad tile reads this
                // lane-octet's token of the ninth quad (token vq of slot
                // TILE_GROUPS) each window; an eight-quad tile runs the loop
                // with no trace of it. Within each, the ninth-token array is
                // written on every path (under a runtime predicate it would
                // go to local memory for the whole kernel — 1.5 KB of spills
                // at the gate).
                if (has8) {
                    const bool tok8 = ((grp_mask(tg.desc[TILE_GROUPS]) >> vq) & 1u) != 0u;
                    #pragma unroll
                    for (int h = 0; h < V_WPW; ++h) {
                        if (vw * V_WPW + h < N_WIN) {
                            float v8[4];
                            tile_quad_vec_token<HEAD_DIM, 2>(ext, tg, s_tbl, 1, TILE_GROUPS,
                                                             win_d0(h), vq, v8);
                            float v[2][4][GROUP_TOK];
                            tile_quads_vec_cvt(raw, 1, v);
                            if (h + 1 < V_WPW && vw * V_WPW + h + 1 < N_WIN)
                                tile_quads_vec_load<HEAD_DIM>(ext, q[h + 1], 1, vgi, raw);
                            v_quantise(std::true_type{}, v, v8, tok8, win_d0(h));
                        }
                    }
                } else {
                    #pragma unroll
                    for (int h = 0; h < V_WPW; ++h) {
                        if (vw * V_WPW + h < N_WIN) {
                            float v[2][4][GROUP_TOK];
                            tile_quads_vec_cvt(raw, 1, v);
                            if (h + 1 < V_WPW && vw * V_WPW + h + 1 < N_WIN)
                                tile_quads_vec_load<HEAD_DIM>(ext, q[h + 1], 1, vgi, raw);
                            const float v8[4] = { 0.f, 0.f, 0.f, 0.f };
                            v_quantise(std::false_type{}, v, v8, false, win_d0(h));
                        }
                    }
                }
            } else {
                // Block path (`tile_v_block_path`): the nine-slot or the
                // eight-slot instantiation under the tile's uniform flag.
                static_assert(TILE_V_WARPS == N_PALETTE, "a V warp per palette");
                static_assert(HEAD_DIM / N_PALETTE == 2 * WARP_SIZE, "two ranks per lane");
                if (has8) tile_v_block_path<HEAD_DIM, TILE_SLOTS>(ext, tg, s_inv, vabs, live_g, vw, lane, s_v8t);
                else      tile_v_block_path<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs, live_g, vw, lane, s_v8t);
                #pragma unroll
                // `a / 127`, as the vector path stores it — not `a · (1/127)`,
                // which rounds differently and would give a multi-pass tile a
                // different PV scale from its single-pass equivalent.
                for (int d = vt; d < HEAD_DIM; d += TILE_V_WARPS * WARP_SIZE)
                    s_v_scale[d] = __float2half(__int_as_float(vabs[d]) / 127.f);
            }
        }
        // Publishes the P and V slabs (and the next tile's descriptors,
        // committed above). Nothing after it touches a slab the next tile
        // writes before its own first barrier, so this is the tile's last
        // barrier.
        __syncthreads();

        // PV: one m16n8k32 per output-dim slice (k = the tile's 32 tokens).
        // The P A-fragment comes from the slab (one ldmatrix), the row
        // rescale factors and the four partial row sums from the softmax
        // warps' posts.
        uint32_t pa[4];
        ldmatrix_x4_b16(pa, sw_a_frag_addr<TILE_TOK>(s_p8, 0, lane));
        float alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            const int r = g + row * 8;
            alpha[row] = s_alpha[r];
            const float4 l4 = *(const float4*)&s_lsum[r][0];
            l_run[row] = l_run[row] * alpha[row] + ((l4.x + l4.y) + (l4.z + l4.w));
        }
        #pragma unroll
        for (int sl8 = 0; sl8 < PV_H; ++sl8) {
            const int dim = dim_base + sl8 * 8;
            uint32_t vb[2];
            ldmatrix_x2_b16(vb, sw_b_frag_addr<TILE_TOK>(s_v8t, dim, 0, lane));
            const float vs0 = __half2float(s_v_scale[dim + n0]) * (1.f / 127.f);
            const float vs1 = __half2float(s_v_scale[dim + n0 + 1]) * (1.f / 127.f);
            int32_t c_i[4] = { 0, 0, 0, 0 };
            int32_t d_i[4];
            mma_int8_m16n8k32(d_i, pa, vb, c_i);
            o_acc[sl8][0] = o_acc[sl8][0] * alpha[0] + (float)d_i[0] * vs0;
            o_acc[sl8][1] = o_acc[sl8][1] * alpha[0] + (float)d_i[1] * vs1;
            o_acc[sl8][2] = o_acc[sl8][2] * alpha[1] + (float)d_i[2] * vs0;
            o_acc[sl8][3] = o_acc[sl8][3] * alpha[1] + (float)d_i[3] * vs1;
        }
    }
    if (!qsa_on || chunk_lo >= e_hi) break;
    // Next sparse chunk: warp 7 resolves it and stages its tile 0, exposed;
    // one barrier publishes both. The last tile's post-softmax barrier
    // orders every thread's reads of s_grp and the descriptor buffers
    // ahead of the rewrite (PV reads only the P and V slabs).
    {
        const int chunk_hi = min(e_hi, chunk_lo + EPS);
        if (warp == STAGE_WARP) {
            const bool has = chunk_lo + lane < chunk_hi;
            const uint32_t ent = has ? sel_entries[chunk_lo + lane] : 0u;
            tile_resolve_entries<HEAD_DIM>(ent, has, sel, slot_idx, slices_ptr,
                                           n_slices, write_slice_idx, n_kv_head, lane,
                                           s_grp, s_grp_src, &s_n_tiles);
            __syncwarp();
            n_tiles = s_n_tiles;
            stage_tile0();
        }
        chunk_lo = chunk_hi;
    }
    __syncthreads();
    n_tiles = s_n_tiles;
    }
    // No tile carried a live token (every selected cell fell outside the
    // slot, or the dense range was all sealed holes): this split is null.
    // Every warp reads the same row sums, so the test is block-uniform.
    if (l_run[0] == 0.f) { emit_null(); return; }

    // ------------------------------------------------------------------
    // Epilogue: this split's un-normalised (ΣpV, m, l) per live row. The
    // softmax warps hold m (log2 units), every warp l; warp 0's quad
    // leaders write both, m back in natural-log units for the combine.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        const int r = g + row * 8;
        if (r >= hpg) continue;
        const int64_t base =
            ((int64_t)slot_idx * n_q_head + first_q_head + r) * num_splits + split_idx;
        float* acc = partial_acc + base * HEAD_DIM;
        #pragma unroll
        for (int sl8 = 0; sl8 < PV_H; ++sl8) {
            const int dim = dim_base + sl8 * 8 + n0;
            *(float2*)&acc[dim] = make_float2(o_acc[sl8][row * 2], o_acc[sl8][row * 2 + 1]);
        }
        if (warp == 0 && (lane & 3) == 0) {
            partial_ml[base * 2] = m_run[row] * TILE_LN2;
            partial_ml[base * 2 + 1] = l_run[row];
        }
    }
}

} // namespace fused_attn
