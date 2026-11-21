// Int8 element helpers, tile constants, group descriptors, and the sparse walk
// that resolves selection entries.
//
// Holds the int8 element helpers, the tile shape and named-barrier constants,
// and the `grp_pack` group descriptor with its accessors.
// Also holds `TileExt`, `TileGroups` and `GroupSrc`, and `tile_slice_holding`
// and `tile_resolve_entries`, which turn one split's selection entries into the
// tile's groups.

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
using int8_elem::i8_pal_rank_quad4;

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
/// palette's arena base, the reciprocal of its palette scale and its format.
/// Index [g][0] = K, [g][1] = V.
///
/// `inv` is `__frcp_rn(scale)`, taken once when the descriptor is staged:
/// every read divides by the palette scale through this reciprocal, and a
/// Q0_V block folds it into its header, so all of them must agree on the
/// correctly rounded value (block_q0_v.cuh) — computing it at each token's
/// read instead cost the refinement and the slow-path call per token.
struct TileExt {
    const char* gbase[TILE_SLOTS][2][N_PALETTE];
    float inv[TILE_SLOTS][2][N_PALETTE];
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
/// stagers the group descriptor plus the palette's pointer, scale
/// reciprocal (`TileExt::inv`) and format. `stage_commit` writes it all to
/// the tile's descriptor buffer.
struct NextTile {
    uint32_t map;
    uint32_t desc;
    int rope0;
    int col0;
    int ngroups;
    uint64_t ptr;
    float inv;
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
