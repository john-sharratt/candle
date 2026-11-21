// Runtime format dispatch, token-window handling, block relevance and K
// threshold scaling, and palette band source resolution.
//
// `with_select_fmt` turns a runtime format code into a compile-time one, and
// the helpers load K, V and Q from the paged arena: the valid token window, the
// Σ(q²k²)/Σ(k²) block relevance, and the per-block K threshold scaled by
// the q-relevance z-score. The band loaders resolve one palette band's source
// view for a (chunk, head) row.

// =============================================================================
// RUNTIME-FMT → COMPILE-TIME-FMT DISPATCH
// =============================================================================
// `with_select_fmt` is the only place the 21-arm fmt switch lives in
// the fused selection kernel. Given a runtime fmt value and a callable
// `f`, it invokes `f(FmtTag<FMT>{})` where FMT is the matching
// SELECT_FMT_* constant. Inside the callable, the tag's `::value` is
// a constexpr, so any call like `search_scales_for_fmt<FMT, HB>(...)`
// resolves at compile time and the body inlines fully.
//
// Usage in the kernel:
//
//     with_select_fmt(fmt, [&](auto tag) {
//         constexpr int FMT = decltype(tag)::value;
//         search_scales_for_fmt<FMT, HB>(is_k, ...);
//     });
//
// Replaces the older macro-based dispatch: the cases are written once
// here, and the call sites read as normal C++ lambdas with no
// preprocessor games. The 21-arm switch costs i-cache once per
// candidate (≤ 6 candidates per slot × 4 slots × 2 sides = ≤ 48 calls
// per kernel block), vs. once per (block, fmt, scale) iteration if the
// switch lived in the inner loop.
template <int V>
struct FmtTag {
    static constexpr int value = V;
};

template <typename F>
__device__ __forceinline__ void with_select_fmt(int fmt, F&& f) {
    switch (fmt) {
        case SELECT_FMT_Q8_KS: f(FmtTag<SELECT_FMT_Q8_KS>{}); return;
        case SELECT_FMT_Q8_0:  f(FmtTag<SELECT_FMT_Q8_0>{});  return;
        case SELECT_FMT_Q8_1:  f(FmtTag<SELECT_FMT_Q8_1>{});  return;
        case SELECT_FMT_Q5_1:  f(FmtTag<SELECT_FMT_Q5_1>{});  return;
        case SELECT_FMT_Q5_0:  f(FmtTag<SELECT_FMT_Q5_0>{});  return;
        case SELECT_FMT_Q4_KS: f(FmtTag<SELECT_FMT_Q4_KS>{}); return;
        case SELECT_FMT_Q4_1:  f(FmtTag<SELECT_FMT_Q4_1>{});  return;
        case SELECT_FMT_Q4_0:  f(FmtTag<SELECT_FMT_Q4_0>{});  return;
        case SELECT_FMT_Q3_1:  f(FmtTag<SELECT_FMT_Q3_1>{});  return;
        case SELECT_FMT_Q3_0:  f(FmtTag<SELECT_FMT_Q3_0>{});  return;
        case SELECT_FMT_Q2_1:  f(FmtTag<SELECT_FMT_Q2_1>{});  return;
        case SELECT_FMT_Q2_0:  f(FmtTag<SELECT_FMT_Q2_0>{});  return;
        case SELECT_FMT_Q2_A:  f(FmtTag<SELECT_FMT_Q2_A>{});  return;
        case SELECT_FMT_Q2_S:  f(FmtTag<SELECT_FMT_Q2_S>{});  return;
        case SELECT_FMT_Q1_S:  f(FmtTag<SELECT_FMT_Q1_S>{});  return;
        case SELECT_FMT_Q0:    f(FmtTag<SELECT_FMT_Q0>{});    return;
        case SELECT_FMT_Q0_V:  f(FmtTag<SELECT_FMT_Q0_V>{});  return;
        case SELECT_FMT_Q1_A:  f(FmtTag<SELECT_FMT_Q1_A>{});  return;
        case SELECT_FMT_Q0_X:  f(FmtTag<SELECT_FMT_Q0_X>{});  return;
        case SELECT_FMT_Q0_M2: f(FmtTag<SELECT_FMT_Q0_M2>{}); return;
        case SELECT_FMT_Q0_M4: f(FmtTag<SELECT_FMT_Q0_M4>{}); return;
        default: return;  // BF16/F16/unknown: not in any candidate ladder
    }
}

// =============================================================================
// CANDIDATE-SET CONTRACT
// =============================================================================
// The host passes per-side `k_candidates` / `v_candidates` arrays —
// SELECT_FMT_* tags ordered by ascending BPE (best compression first).
// `search_scales_for_fmt` walks them in that order and stops at the
// first format that hits the HB/N_PALETTE-block slot quota; the result is
// the most aggressive format the slot can sustain. Within an equal-BPE tier
// the search picks the lowest-aerr format. If no candidate hits the quota,
// the fallback path uses the highest-BPE candidate seen, with its best
// scale and (partial) pass mask.
//
// Kept here for ABI parity with the auxiliary sample/winner/reduce
// kernels below, which were already 4-warp.
#define SELECT_WARPS_PER_BLOCK 4

// =============================================================================
// INPUT DTYPE LOAD HELPER
// =============================================================================
// Loads element `idx` from a float-typed arena (F32, F16, or BF16). The
// dtype code is per arena and resolved by the caller from the per-head
// table entry. R16 is a separate block-structured format that carries
// Q activations alongside K values and is handled via `dequant_element`
// + `dequant_q_element`, not this helper.
#define SELECT_INPUT_F32  0
#define SELECT_INPUT_F16  1
#define SELECT_INPUT_BF16 2
#define SELECT_INPUT_R16  3   // K stored as block_r16 (d[32]=K, q[32]=Q); V still float

__device__ __forceinline__ float load_as_float(const void* __restrict__ data, int idx, int input_dtype) {
    if (input_dtype == SELECT_INPUT_F16) {
        return __half2float(((const __half*)data)[idx]);
    } else if (input_dtype == SELECT_INPUT_BF16) {
        return __bfloat162float(((const __nv_bfloat16*)data)[idx]);
    } else {
        return ((const float*)data)[idx];
    }
}

// =============================================================================
// VALID TOKEN WINDOW
// =============================================================================
// A chunk's live tokens are [lo, hi), unpacked from `(offset << 8) | len`.
// Every load of chunk data below reads a token outside the window as ZERO.
// A partial chunk's dead slots hold whatever the ground held before the
// chunk was claimed — the pool does not clear a claimed slot, and a recycled
// region is not cleared either — so a stale inf there would otherwise set the
// head amax, a block amax and every candidate scale for the live tokens.
//
// Every load is arranged so the lane is the token. A quantized / R16 band is
// token-oriented (block `bib` is one dim's 32 tokens); a float band is
// token-major `[t][pd]` and is read at `lane * band_dims + bib`, dim `bib`'s
// value for token `lane`. Block `bib` is therefore dim `bib`'s 32 tokens in
// either layout — the block the convert encodes and the palette map assigns.
struct TokenWindow { int lo; int hi; };

__device__ __forceinline__ TokenWindow unpack_token_window(int packed) {
    const int lo  = (packed >> 8) & 0xff;
    const int len = max(1, min(32, packed & 0xff));
    return { lo, lo + len };
}

__device__ __forceinline__ bool lane_in_window(int lane, TokenWindow w) {
    return lane >= w.lo && lane < w.hi;
}

// Flat index, in a token-major float band `band_dims` wide, of dim `dim`'s
// value for token `lane`.
__device__ __forceinline__ int float_band_elem(int dim, int lane, int band_dims) {
    return lane * band_dims + dim;
}

// =============================================================================
// BLOCK RELEVANCE  —  Σ(q²k²) / Σ(k²)  per block
// =============================================================================
// Coarse proxy for how much the query attends to a given K block.
//
// Derivation: the attention logit for an element pair is q·k. Squaring
// and summing across the 32 lanes of a block gives a non-negative
// magnitude signal; normalising by Σ k² removes the trivial dependence
// on the block's own amax, leaving a (block, query)-conditioned weight
// that's:
//
//   - large when q and k are correlated and aligned in the dominant lanes,
//   - small when q is approximately orthogonal to k or its energy lives
//     in lanes where k is small.
//
// Used in two places:
//
//   - K side: per-block `kthresh[b]` is z-scaled by q-relevance via
//     `k_threshold_scaled`. High relevance → tighter threshold (preserve
//     precision where it matters); low relevance → looser threshold.
//   - Phase 2.5: per-token sink detection uses the same per-element
//     q·k pattern, but at the per-token (not per-block) level.
//
// Q activations are only available when the source K block is in the
// R16 format; for other source formats `q_val == 0` everywhere, has_q
// is false, and the q-relevance path collapses (qrel = 1.0, kthresh
// falls back to sqrt(lo·hi)).

__device__ __forceinline__ float per_lane_qk2(float k, float q) {
    return __fmul_rn(__fmul_rn(q, q), __fmul_rn(k, k));
}

// Warp-reduce per-lane qk² (and k²) into the block relevance scalar.
// Returns Σ(q²k²) / Σ(k²) on every lane after the broadcast.
__device__ __forceinline__ float block_relevance_from_qk2(float qk2_lane, float k) {
    float qk2 = qk2_lane;
    float k2  = __fmul_rn(k, k);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        qk2 += __shfl_xor_sync(0xffffffff, qk2, offset, 32);
        k2  += __shfl_xor_sync(0xffffffff, k2,  offset, 32);
    }
    if (k2 == 0.0f) return 0.0f;
    return __fdiv_rn(qk2, k2);
}

// Convenience wrapper. Identical to qk2 → block_relevance_from_qk2,
// but retains the (k, q) signature where lanes pass their raw values.
__device__ __forceinline__ float block_relevance(float k, float q) {
    return block_relevance_from_qk2(per_lane_qk2(k, q), k);
}

// =============================================================================
// K THRESHOLD SCALING  —  per-block threshold from q-relevance z-score
// =============================================================================
// Anchor at sqrt(lo·hi) (the geometric mean), then scale by a
// z-conditioned multiplier:
//
//     z          = (q_relevance − q_median) / max(q_spread, 1e-8)
//     multiplier = exp(−z)
//     scaled     = sqrt(lo · hi) · multiplier
//     kthresh[b] = clamp(scaled, threshold_hi, threshold_lo)
//
// High q-relevance (z > 0) → smaller threshold (stricter on blocks the
// query attends to); low q-relevance (z < 0) → larger threshold (more
// permissive). The exp(−z) shape gives smooth, monotonic interpolation
// without the saturation kinks of a piecewise lerp.
//
// Convention: `threshold_lo > threshold_hi` numerically. lo = lenient
// (applied at low relevance), hi = strict (applied at high relevance).
// The clamp is anti-symmetric in lo/hi to make the bounds explicit.
//
// q_median and q_spread are precomputed per (chunk, head) by Pass 1
// (`approximate_q_relevance_quantiles`); spread is IQR-scaled so the
// z-score is approximately standard-normal regardless of the block's
// q-relevance distribution.
__device__ __forceinline__ float k_threshold_scaled(
    float threshold_lo,
    float threshold_hi,
    float q_relevance,
    float q_median,
    float q_spread
) {
    const float safe_spread = fmaxf(q_spread, 1.0e-8f);
    const float z           = __fdiv_rn(q_relevance - q_median, safe_spread);
    const float multiplier  = __expf(-z);
    const float base        = sqrtf(__fmul_rn(threshold_lo, threshold_hi));
    const float scaled      = __fmul_rn(base, multiplier);
    return fmaxf(threshold_hi, fminf(threshold_lo, scaled));
}


// =============================================================================
// ARENA HELPERS  —  paged KV-cache addressing
// =============================================================================
// The KV cache is paged: each (chunk, head) pair has its data stored
// inside an "arena" (a large per-arena contiguous buffer), addressed
// via per-head table entries that resolve to byte offsets and strides.
// These helpers are the small primitives used to walk that addressing
// scheme — invoked in tight loops over (chunk, head) inside every
// kernel in this file. Layout matches `paged_decode_kernel` exactly so
// the selection kernel sees the same bytes the attention kernel will:
//
//   - Palette4PerHeadEntry (36 × i64): one row per (chunk, kv_head),
//     indexed `chunk_idx * n_kv_head + head_idx`, holding four palette
//     sub-entries. Each sub-entry describes ONE band: its own base
//     pointer, byte offset, slot stride, format tag and outer scale.
//   - PerHeadTableEntry (9 × i64): one such sub-entry.
//   - head_gids: per-chunk per-head K/V global chunk IDs (interleaved).
//   - The gid supplies only the band's slot index within its region:
//        chunk_in_region = gid % arena_chunks
//     The row key comes from `chunk_idx`, not from the gid, so bands of
//     one head may sit in different regions and different formats.
//   - load_band_entry → per_head_k_ptr / per_head_v_ptr for byte-level
//     addressing.
//   - Format tags come from per-band metadata, not a global dtype param.

// Float arena formats in `ArenaFormat::*` map directly onto the
// `SELECT_INPUT_*` codes used by `load_as_float`: F32=0, F16=1, BF16=2.
// Quantized formats are handled separately via `dequant_element_inline`.
__device__ __forceinline__ int arena_fmt_to_dtype_code(int fmt) {
    return fmt;  // ArenaFormat::F32=0, F16=1, BF16=2
}

// Byte size of a single 32-element block for the given ArenaFormat.
// Used to step through a chunk's quantized data block-by-block.
__device__ __forceinline__ int quant_block_bytes(int fmt) {
    switch (fmt) {
        case ArenaFormat::F16:  return 64;
        case ArenaFormat::R16:  return 128;
        case ArenaFormat::Q8_KS: return 36;
        case ArenaFormat::Q8_1: return 36;
        case ArenaFormat::Q8_0: return 34;
        case ArenaFormat::Q5_1: return 24;
        case ArenaFormat::Q5_0: return 22;
        case ArenaFormat::Q4_KS: return 20;
        case ArenaFormat::Q4_1: return 20;
        case ArenaFormat::Q4_0: return 18;
        case ArenaFormat::Q3_1: return 16;
        case ArenaFormat::Q3_0: return 14;
        case ArenaFormat::Q2_0: return 10;
        case ArenaFormat::Q2_1: return 12;
        case ArenaFormat::Q2_A: return 10;
        case ArenaFormat::Q2_S: return 9;
        case ArenaFormat::Q0_M4: return 8;
        case ArenaFormat::Q1_S: return 5;
        case ArenaFormat::Q0_M2: return 3;
        case ArenaFormat::Q0_V: return 2;
        case ArenaFormat::Q1_A: return 6;
        case ArenaFormat::Q0_X: return 2;
        case ArenaFormat::Q0: return 1;
        // Non-arena format reached arena block addressing — fail loud rather than
        // mis-address with a guessed size. K-quants/AWQ/q8a128 use non-32-element
        // blocks and are not arena-storable, so this is always a programming error.
        default: __trap(); return 0;
    }
}

// Extract the Q-capture value at element index `idx` from a quantized
// block. Only R16 carries Q (in the `q[]` field); for any other format
// the compiler folds this to a constant 0.0f, so the call is zero-cost
// on non-R16 paths. Pass 1 uses this to decide whether q-relevance is
// usable for the head; the main kernel uses it to populate q_vals_half
// (Phase 1) which is then mean-reduced into `q_mean` for Phase 2.5
// sink detection.
__device__ __forceinline__ float dequant_q_element(const void* block_ptr, int idx, int fmt) {
    if (fmt == ArenaFormat::R16) {
        const block_r16* blk = reinterpret_cast<const block_r16*>(block_ptr);
        return __half2float(*reinterpret_cast<const __half*>(&blk->q[idx]));
    }
    return 0.0f;
}

// Look up one palette band's sub-entry from the (chunk, head) row of the
// selection table. Bridges the raw int64 buffer the host passes in to the
// typed `Palette4PerHeadEntry` view.
//
// Rows are keyed by `chunk_idx * n_kv_head + head_idx`, and each of the four
// sub-entries describes that band alone — its own base pointer, slot stride,
// format tag and outer scale. Keying by arena instead would force every band
// of a head to share one arena and one format, which is exactly what size
// classes remove (see `docs/arena_unification.md` §2).
__device__ __forceinline__ PerHeadTableEntry load_band_entry(
    const int64_t* __restrict__ per_head_table_raw,
    int chunk_idx,
    int head_idx,
    int n_kv_head,
    int palette
) {
    const Palette4PerHeadEntry* per_head_table =
        reinterpret_cast<const Palette4PerHeadEntry*>(per_head_table_raw);
    return palette4_sub_entry(per_head_table[chunk_idx * n_kv_head + head_idx], palette);
}

// Resolve one palette band's source view (chunk base pointer, storage format,
// outer scale) from that band's OWN gid and its OWN table sub-entry.
//
// Each (head, palette) band is an independent arena slot: bands of one head
// may live in different arenas, at non-contiguous chunk indices, and in
// different formats (format selection is per-(head, palette)). The selection
// kernels therefore address every band through its own gid; nothing here
// assumes band contiguity or format uniformity across a head.
//
// The gid supplies only the band's *chunk index within its region*; the base
// pointer, stride and format come from the (chunk, head) row's palette
// sub-entry, so a region shared by several formats resolves each band
// correctly.
//
// The head_gids buffer carries all GIDS_PER_HEAD gids per (chunk, head), in
// HeadGids order: head * GIDS_PER_HEAD + palette * 2 + is_v.
__device__ __forceinline__ void resolve_band_source(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    int chunk_idx,
    int head_idx,
    int n_kv_head,
    int arena_chunks,
    int palette,
    bool is_v,
    const char** ptr_out,
    int* fmt_out,
    float* outer_out
) {
    const int64_t gid = __ldg(&head_gids[
        (int64_t)chunk_idx * n_kv_head * GIDS_PER_HEAD
        + head_idx * GIDS_PER_HEAD + palette * 2 + (is_v ? 1 : 0)]);
    const int arena_idx      = (int)(gid / (int64_t)arena_chunks);
    const int chunk_in_arena = (int)(gid - (int64_t)arena_idx * (int64_t)arena_chunks);
    PerHeadTableEntry e = load_band_entry(per_head_table_raw, chunk_idx, head_idx, n_kv_head, palette);
    if (is_v) {
        *ptr_out   = per_head_v_ptr(e) + (int64_t)chunk_in_arena * e.v_chunk_byte_stride;
        *fmt_out   = per_head_get_v_format(e);
        *outer_out = per_head_get_v_scale(e);
    } else {
        *ptr_out   = per_head_k_ptr(e) + (int64_t)chunk_in_arena * e.k_chunk_byte_stride;
        *fmt_out   = per_head_get_k_format(e);
        *outer_out = per_head_get_k_scale(e);
    }
}

// Lane `lane`'s K (and Q, when WITH_Q) and V values of LOAD_GROUP blocks of one
// head, `g`, `g + stride`, …, all inside one band, read through that band's
// source views (`band_*[0]` K, `band_*[1]` V). Every load is issued before any
// value is used, so a warp waits out one memory latency per group rather than
// one per block. Q is zero where the K source does not carry it. Values are as
// read, before any token-window masking.
template <int LOAD_GROUP, bool WITH_Q>
__device__ __forceinline__ void load_band_group(
    const char* const (*band_ptr)[N_PALETTE],
    const int   (*band_fmt)[N_PALETTE],
    const float (*band_outer)[N_PALETTE],
    int g, int stride, int band_blocks, int lane,
    float (&k)[LOAD_GROUP], float (&q)[LOAD_GROUP], float (&v)[LOAD_GROUP])
{
    const int p = g / band_blocks;
    const int   k_fmt  = band_fmt[0][p];
    const char* k_data = band_ptr[0][p];
    const int   v_fmt  = band_fmt[1][p];
    const char* v_data = band_ptr[1][p];
    #pragma unroll
    for (int j = 0; j < LOAD_GROUP; j++) {
        const int bib = g + j * stride - p * band_blocks; // block within band
        q[j] = 0.0f;
        if (ArenaFormat::is_quantized(k_fmt)) {
            const char* k_blk = k_data + (int64_t)bib * quant_block_bytes(k_fmt);
            k[j] = dequant_element_inline<float, true>(k_blk, lane, k_fmt, band_outer[0][p]);
            if constexpr (WITH_Q) q[j] = dequant_q_element(k_blk, lane, k_fmt);
        } else {
            k[j] = load_as_float(k_data, float_band_elem(bib, lane, band_blocks), arena_fmt_to_dtype_code(k_fmt));
        }
        if (ArenaFormat::is_quantized(v_fmt)) {
            const char* v_blk = v_data + (int64_t)bib * quant_block_bytes(v_fmt);
            v[j] = dequant_element_inline<float, false>(v_blk, lane, v_fmt, band_outer[1][p]);
        } else {
            v[j] = load_as_float(v_data, float_band_elem(bib, lane, band_blocks), arena_fmt_to_dtype_code(v_fmt));
        }
    }
}
