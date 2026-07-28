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
 *    loaded once per group instead of once per head-block. M_ROWS = 64
 *    (4 m16 row-tiles, each served by a warp PAIR under the head-dim
 *    split), BLOCK_M_TOK = 64 / hpg tokens.
 *
 *  - SLICE-ALIGNED TILES: a KV tile is one TokenSlice run (≤ 32 tokens,
 *    never straddling a chunk). One palette table per tile; the straddle
 *    twin-table scheme of the FP16 kernel is structurally unnecessary.
 *
 *  - RAW-FIRST STAGING: each palette's 32-token quant-block span is
 *    bulk-copied to smem with 16-byte cp.async (perfectly coalesced),
 *    and every per-element decode — K dequant→RoPE→requant, V int8
 *    read-through or FP requant — extracts from that smem copy in
 *    natural dim order via the rank tables. There is no FP16 exchange
 *    slab: the rank→natural permutation happens in the table-indexed
 *    reads themselves. Dtype palettes (unsealed float prefixes) stage
 *    their raw element spans the same way, K and V phased sequentially
 *    through one scratch region. Non-hop palettes (R16, F32, unaligned
 *    spans) decode element-wise straight from global — rare.
 *
 *  - FRESH TOKENS FROM THE INPUTS: the q_len new tokens are staged straight
 *    from the packed q/k/v tensors (never read back from the arena); the
 *    arena write of their K/V is an independent pre-pass (z == 0 only).
 *
 * Quantization grid (independent of the arena's palette routing):
 *    Q:  int8 per (M-row, 32-dim window)   — natural dim order
 *    K:  int8 per (token, 32-dim window)   — natural dim order, post-RoPE
 *    P:  int8 per row, fixed scale 1/127   (P ∈ (0, 1] after online softmax)
 *    V:  int8 per (natural dim, tile)      — arena block scale (read-through)
 *                                            or requant max-abs (fallback)
 *  QK epilogue: acc_f32 += i32(window) · qs[row][w] · ks[tok][w]
 *  PV epilogue: o_f32   += i32 · (1/127) · vs[dim]
 *  The O accumulator and V^T slab are NATURAL-dim indexed — palette rank
 *  space is per-slice and cannot host a cross-tile accumulator.
 *
 * v1 scope: HEAD_DIM % 64 == 0 (in-thread RoPE pairing). Read-through V
 * engages whenever every V palette's format is an int8 passthrough family
 * (per-element extraction has no lane-width constraint); asymmetric or
 * dtype V palettes take the FP-fallback path.
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
#include "../arena_table.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../convert/convert_all.cuh"
#include "../mma/mma_wrappers.cuh"
#include "pal_rank.cuh"
#include "kv_store.cuh"

namespace prefill_int8 {

using fused_attn::load_a_frag_m16k32_ldmatrix;
using fused_attn::load_b_frag_n8k32_ldmatrix;
using fused_attn::mma_int8_m16n8k32;

constexpr int I8_WARPS = 8;
constexpr int I8_THREADS = I8_WARPS * 32;
// Head-dim-split warp pairing: warp = (row-tile, dim-half). Each PAIR of
// warps serves one m16 row-tile — both duplicate the (cheap) QK + softmax
// for those 16 rows, and each accumulates only HALF the output dims. That
// halves the o_acc register hog (64 → 32 FP32/thread), which is what
// makes the 64-register budget of the 4-blocks/SM max-occupancy
// configuration reachable. Staging is unchanged — K and V^T are staged
// once per block and shared through smem, so no global traffic is
// duplicated. The known cost: compute-bound shapes (long q, short
// prefix) pay for the duplicated QK (§13.3 rounds 6–9).
constexpr int I8_ROW_TILES = I8_WARPS / 2;
constexpr int I8_M_ROWS = I8_ROW_TILES * 16; // 64 M-rows per block
constexpr int I8_TILE_TOK = 32;              // one chunk-slice per tile

/// cos/sin lookup, same table layout as the FP16 kernel:
/// rope_cs[pos*HD + 2i] = cos, [.. + 2i + 1] = sin for pair (i, i + HD/2).
template <int HEAD_DIM>
__device__ __forceinline__ void i8_rope_cs(
    int pos, int d_idx, const float* __restrict__ rope_cs, float& c, float& s)
{
    const float* e = rope_cs + (int64_t)pos * HEAD_DIM + d_idx * 2;
    c = __ldg(e);
    s = __ldg(e + 1);
}

template <typename QT>
__device__ __forceinline__ float qt_to_f32(QT v);
template <>
__device__ __forceinline__ float qt_to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float qt_to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename QT>
__device__ __forceinline__ QT qt_from_f32(float v);
template <>
__device__ __forceinline__ __half qt_from_f32<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 qt_from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

/// cp.async fences for the raw-block staging fill. Groups are per-thread:
/// every thread commits and drains its own copies before the block-wide
/// staging barrier makes them visible (a bare __syncthreads does NOT
/// fence cp.async).
__device__ __forceinline__ void i8_cp_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

__device__ __forceinline__ void i8_cp_wait0() {
    asm volatile("cp.async.wait_group 0;" ::);
}

/// 16-byte global→shared bulk copy (both pointers 16-byte aligned).
__device__ __forceinline__ void i8_cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                    "l"(src));
}

/// Quantize a value against a precomputed window scale (0 ⇒ all-zero window).
__device__ __forceinline__ int8_t i8_quant(float v, float inv_scale) {
    float q = rintf(v * inv_scale);
    q = fminf(127.f, fmaxf(-127.f, q));
    return (int8_t)q;
}

/// Runtime-format single-element FP decode from a token-oriented quant
/// block (`blk` points at ONE dim's block; `e` is the token within it).
/// Same numerics as load_head_quant_token_oriented: value / scale.
__device__ __forceinline__ float i8_dequant_elem(
    int fmt, const char* blk, int e, float scale)
{
    switch (fmt) {
#define I8_DQ(F, B) \
    case ArenaFormat::F: \
        return BlockConverter<B, float>::load_element((const B*)blk, e, scale)
        I8_DQ(R16, block_r16);
        I8_DQ(Q4_0, block_q4_0);
        I8_DQ(Q4_1, block_q4_1);
        I8_DQ(Q5_0, block_q5_0);
        I8_DQ(Q5_1, block_q5_1);
        I8_DQ(Q8_0, block_q8_0);
        I8_DQ(Q8_1, block_q8_1);
        I8_DQ(Q4_KS, block_q4_ks);
        I8_DQ(Q8_KS, block_q8_ks);
        I8_DQ(Q3_0, block_q3_0);
        I8_DQ(Q3_1, block_q3_1);
        I8_DQ(Q2_0, block_q2_0);
        I8_DQ(Q2_1, block_q2_1);
        I8_DQ(Q2_A, block_q2_a);
        I8_DQ(Q2_S, block_q2_s);
        I8_DQ(Q1_S, block_q1_s);
        I8_DQ(Q0, block_q0);
        I8_DQ(Q0_V, block_q0_v);
        I8_DQ(Q1_A, block_q1_a);
        I8_DQ(Q0_X, block_q0_x);
        I8_DQ(Q0_M2, block_q0_m2);
        I8_DQ(Q0_M4, block_q0_m4);
#undef I8_DQ
        // A non-arena format reached block extraction — fail loud (the
        // same __trap idiom as the accessor's block addressing).
        default: __trap(); return 0.f;
    }
}

/// Runtime-format single-element int8 read-through (V). Same families and
/// numerics as load_head_int8_readthrough's dispatcher.
__device__ __forceinline__ Int8Sample i8_rt_elem(int fmt, const char* blk, int e)
{
    switch (fmt) {
#define I8_RT(F, B) \
    case ArenaFormat::F: return BlockInt8<B>::load((const B*)blk, e)
        I8_RT(Q8_0, block_q8_0);
        I8_RT(Q4_0, block_q4_0);
        I8_RT(Q5_0, block_q5_0);
        I8_RT(Q2_0, block_q2_0);
        I8_RT(Q3_0, block_q3_0);
        I8_RT(Q4_KS, block_q4_ks);
        I8_RT(Q8_KS, block_q8_ks);
        I8_RT(Q8_1, block_q8_1);
        I8_RT(Q2_S, block_q2_s);
        I8_RT(Q1_S, block_q1_s);
        I8_RT(Q1_A, block_q1_a);
        I8_RT(Q0, block_q0);
        I8_RT(Q0_M2, block_q0_m2);
        I8_RT(Q0_M4, block_q0_m4);
        I8_RT(Q0_X, block_q0_x);
#undef I8_RT
        default: __trap(); return Int8Sample{0, 0.f};
    }
}

/// One arena element as FP32, from either a quant-block span (bb > 0:
/// `base` is the palette's raw smem copy or its global span) or a
/// channel-oriented dtype palette (bb == 0: element addressing).
/// Matches load_head_scaled's semantics: decoded value / scale (the
/// dtype identity fast path skips the divide only when scale == 1.0f,
/// where /1.0f is exact anyway).
__device__ __forceinline__ float i8_arena_elem(
    int fmt, int bb, const char* base, int rank, int within, float scale, int sub)
{
    if (bb > 0)
        return i8_dequant_elem(fmt, base + (int64_t)rank * bb, within, scale);
    const int es = ArenaFormat::float_elem_size(fmt);
    const char* pe = base + ((int64_t)within * sub + rank) * es;
    float v;
    if (fmt == ArenaFormat::F16) {
        v = __half2float(*(const __half*)pe);
    } else if (fmt == ArenaFormat::BF16) {
        v = __bfloat162float(*(const __nv_bfloat16*)pe);
    } else if (fmt == ArenaFormat::F32) {
        v = *(const float*)pe;
    } else { // F8E4M3
        v = to_float<__nv_fp8_e4m3>(*(const __nv_fp8_e4m3*)pe);
    }
    return v / scale;
}

// ============================================================================
// The kernel
// ============================================================================

// minBlocks = 4 pins the register budget at 64/thread — 4 blocks × 256
// threads is the max-occupancy configuration (67% theoretical; 6 blocks
// would need ≤42 regs, unreachable past o_acc's 32). The Q-fragment
// drain, the union smem arena, and the deliberately register-lean
// staging (recompute lambdas, smem-resident Q scales, two-pass V
// requant, serialized palette loops) are what make the budget close
// with only a small residual spill.
template <typename QT, int HEAD_DIM>
__global__ void __launch_bounds__(I8_THREADS, 4)
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
    int rope_interleaved,               // 0 only in v1 (asserted host-side)
    // Split-KV: grid.z = batch_size × num_splits. Shard s of a sequence
    // processes tiles (sealed AND fresh — one shared ordinal space) with
    // ordinal ≡ s (mod num_splits). num_splits == 1 stores O directly;
    // otherwise each shard emits
    // an un-normalized (ΣpV, m, l) partial into `partials`
    // [total_q·n_head rows][num_splits][HEAD_DIM + 2] and the combine kernel
    // merges them (base-e log-sum-exp).
    int num_splits,
    float* __restrict__ partials
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 prefill: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;       // QK k-step windows (also dims/lane)
    constexpr int PV_SLICES = HEAD_DIM / 8;    // PV n-slices (output dims per mma)
    constexpr int SUB = HEAD_DIM / N_PALETTE;  // palette band width
    // The rank tables pack (palette, rank) into one byte as p<<6 | rank,
    // and the raw-span fill copies in 16-byte units.
    static_assert(N_PALETTE == 4, "rank-table byte packs the palette into 2 bits");
    static_assert(SUB >= 16 && SUB <= 64,
                  "rank needs 6 bits; raw spans copy in 16-byte units");

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_tile = warp >> 1; // which m16 row-tile this warp serves
    const int dim_half = warp & 1;  // which output-dim half it accumulates
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
    //     (s_k8/s_v8t + scales), the raw∪p8∪fresh scratch, and the palette
    //     tables. These cannot union among themselves — the slabs span the
    //     staging barrier and the tables persist across tiles — but all of
    //     them may alias the dead prologue.
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
    // Raw staging spans: one 32-token block run per palette per side,
    // sized for the largest arena block (Q8_1/Q8_KS = 36 B/dim). SUB ≥ 16
    // keeps every slot a multiple of 16 bytes (cp.async-aligned).
    constexpr int RAW_SLOT = SUB * 36;
    constexpr int RAW_BYTES = N_PALETTE * RAW_SLOT;
    // Dtype-tile slot: a float palette span is 32 tokens × SUB dims × 2 B
    // (F16/BF16; F8 is half that, F32 does not fit and stays non-hop).
    // Both sides cannot fit simultaneously, so dtype tiles stage K and V
    // SEQUENTIALLY through one N_PALETTE × RAW_SLOT_D region (see the
    // staging phase below).
    constexpr int RAW_SLOT_D = I8_TILE_TOK * SUB * 2;
    constexpr int FRESH_BYTES = I8_TILE_TOK * HEAD_DIM * 2;
    constexpr int SCRATCH_BYTES =
        (2 * RAW_BYTES > P8_BYTES)
            ? ((2 * RAW_BYTES > FRESH_BYTES) ? 2 * RAW_BYTES : FRESH_BYTES)
            : ((P8_BYTES > FRESH_BYTES) ? P8_BYTES : FRESH_BYTES);
    static_assert(N_PALETTE * RAW_SLOT_D <= SCRATCH_BYTES,
                  "one side's dtype spans must fit the scratch region");

    constexpr int ALIGN16 = 15;
    // Tile overlay offsets (all 16-aligned).
    constexpr int OFF_K8 = 0;
    constexpr int OFF_KS = (OFF_K8 + I8_TILE_TOK * Q8_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_V8T = (OFF_KS + I8_TILE_TOK * N_WIN * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VS = (OFF_V8T + HEAD_DIM * V8T_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_SCR = (OFF_VS + HEAD_DIM * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_TBLK = (OFF_SCR + SCRATCH_BYTES + ALIGN16) & ~ALIGN16;
    constexpr int OFF_TBLV = OFF_TBLK + HEAD_DIM;
    constexpr int TILE_BYTES = OFF_TBLV + HEAD_DIM;
    // Prologue overlay (s_q8 only — the Q scales are RESIDENT, below).
    constexpr int PRO_BYTES = I8_M_ROWS * Q8_LD;
    constexpr int ARENA_BYTES = (TILE_BYTES > PRO_BYTES) ? TILE_BYTES : PRO_BYTES;
    static_assert(ARENA_BYTES + 128 <= 25600,
                  "arena must fit the 4-blocks/SM smem budget (25.6 KB)");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    __shared__ uint8_t s_pal_cache[HEAD_DIM / 2];
    __shared__ int s_tbl_valid;
    // Q scales stay RESIDENT in smem (512 B of the 4-block headroom): the
    // QK fixup reads them as broadcasts, and NOT draining them to
    // registers hands ptxas 4 regs/thread of slack at the 64-reg cap —
    // measured spill traffic was ~25% of global sector volume.
    __shared__ __half s_q_scale[I8_M_ROWS][N_WIN];
    // Per-palette extraction metadata for the current tile (resident —
    // 160 B): decode base (raw smem copy, or global for non-hop
    // palettes), palette scale, format, and quant block bytes (0 ⇒ dtype
    // element addressing). Index [0] = K, [1] = V. Block-uniform values;
    // reads are smem broadcasts.
    __shared__ const char* s_ext_base[2][N_PALETTE];
    __shared__ float s_ext_scl[2][N_PALETTE];
    __shared__ int s_ext_fmt[2][N_PALETTE];
    __shared__ int s_ext_bb[2][N_PALETTE];

    // Prologue view (dead after the Q drain barrier).
    auto s_q8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena);
    // Tile views (alias the prologue bytes — valid only after the drain).
    auto s_k8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena + OFF_K8);
    auto s_k_scale = reinterpret_cast<__half(*)[N_WIN]>(s_arena + OFF_KS);
    auto s_v8t = reinterpret_cast<int8_t(*)[V8T_LD]>(s_arena + OFF_V8T);
    auto s_v_scale = reinterpret_cast<__half*>(s_arena + OFF_VS);
    uint8_t* s_tbl_k = s_arena + OFF_TBLK;
    uint8_t* s_tbl_v = s_arena + OFF_TBLV;

    // Scratch tenant views (inside the tile overlay). Temporally disjoint:
    //   s_raw_k/v — bulk-copied raw quant-block spans, SEALED staging only.
    //   s_fresh   — FP16 V stash for the per-dim requant, FRESH staging only.
    //   s_p8      — per-warp quantized P tiles, COMPUTE phase only.
    char* s_raw_k = (char*)(s_arena + OFF_SCR);
    char* s_raw_v = s_raw_k + RAW_BYTES;
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
            #pragma unroll
            for (int w = 0; w < N_WIN / 2; ++w) {
                int d = lane + 32 * w;
                float c, s;
                i8_rope_cs<HEAD_DIM>(pos, d, rope_cs, c, s);
                float lo = x[w], hi = x[w + N_WIN / 2];
                x[w] = lo * c - hi * s;
                x[w + N_WIN / 2] = lo * s + hi * c;
            }
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
            // Slot-state integrity guard for the write path (see the
            // sealed-tile guard): a wild slice index or null record here
            // would WRITE through a garbage pointer. Name the values and skip
            // the token. Block-uniform values — no barrier divergence.
            if (w_slice >= (int)slot_hdr.n_slices || w_in_blk >= 32) {
                if (tid == 0) {
                    printf("PPI8 GUARD(W): pos %d of slot(b=%d,h=%d) -> slice %d/%u in_blk %d "
                           "(prefix=%d q=%d)\n",
                           prefix_len + tok, batch_idx, kv_head_idx, w_slice,
                           slot_hdr.n_slices, w_in_blk, prefix_len, q_len);
                }
                continue;
            }
            const uint8_t* w_sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, w_slice, n_kv_head);
            const uint64_t w_rec = *reinterpret_cast<const uint64_t*>(w_sl + 8);
            if (w_rec == 0) {
                if (tid == 0) {
                    printf("PPI8 GUARD(W): null kvheads_ptr at pos %d slice %d of slot(b=%d,h=%d)\n",
                           prefix_len + tok, w_slice, batch_idx, kv_head_idx);
                }
                continue;
            }
            const uint8_t* w_head = get_head<HEAD_DIM>(w_sl, kv_head_idx);
            bool w_bands_ok = true;
            #pragma unroll
            for (int p = 0; p < N_PALETTE; ++p) {
                w_bands_ok = w_bands_ok && kvhead_k_ptr<HEAD_DIM>(w_head, p) != 0
                                        && kvhead_v_ptr<HEAD_DIM>(w_head, p) != 0;
            }
            if (!w_bands_ok) {
                if (tid == 0) {
                    printf("PPI8 GUARD(W): null band ptr in writer record at pos %d slice %d "
                           "of slot(b=%d,h=%d)\n",
                           prefix_len + tok, w_slice, batch_idx, kv_head_idx);
                }
                continue;
            }
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
    if (tid == 0) s_tbl_valid = 0;
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
    // warp's dim-half; its dims start at dim_half * (HEAD_DIM / 2).
    // ------------------------------------------------------------------
    constexpr int PV_H = PV_SLICES / 2;
    const int dim_base = dim_half * (HEAD_DIM / 2);
    float o_acc[PV_H][4];
    #pragma unroll
    for (int s = 0; s < PV_H; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    // ==================================================================
    // Tile loop: sealed slices (logical [0, prefix_len)), then fresh
    // 32-token tiles (logical [prefix_len, kv_len), staged from inputs).
    // ==================================================================
    int cur = 0;
    int tile_ord = 0; // tile ordinal (sealed and fresh), for split-KV round-robin
    while (cur < kv_len) {
        const bool fresh = (cur >= prefix_len);
        int tile_len;
        int in_blk0 = 0;
        const uint8_t* sl_head = nullptr;

        if (!fresh) {
            int sl_idx;
            resolve_pos(slot_hdr, cur, sl_idx, in_blk0);
            // Slot-state integrity guard: a resolved position must land inside
            // the slot's slice array with an in-chunk offset < 32, and the
            // slice's record and band pointers must be non-null. A violation
            // means the host staged inconsistent slot state — the known class
            // is a member whose logical offset ran ahead of its physical
            // backing (kv_len spans positions the block table doesn't cover),
            // which sends this resolve past the slot's span in the packed
            // uploads. Dereferencing would be a wild read with no attribution;
            // name the values and skip the position instead. All values are
            // block-uniform, so the branch cannot diverge the barriers.
            if (sl_idx >= (int)slot_hdr.n_slices || in_blk0 >= 32) {
                if (tid == 0) {
                    printf("PPI8 GUARD: pos %d of slot(b=%d,h=%d) -> slice %d/%u in_blk %d "
                           "(prefix=%d q=%d kv=%d)\n",
                           cur, batch_idx, kv_head_idx, sl_idx, slot_hdr.n_slices, in_blk0,
                           prefix_len, q_len, kv_len);
                }
                cur += 1;
                continue;
            }
            const uint8_t* sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, sl_idx, n_kv_head);
            const uint64_t rec_ptr = *reinterpret_cast<const uint64_t*>(sl + 8);
            if (rec_ptr == 0) {
                if (tid == 0) {
                    printf("PPI8 GUARD: null kvheads_ptr at pos %d slice %d of slot(b=%d,h=%d) "
                           "(prefix=%d q=%d)\n",
                           cur, sl_idx, batch_idx, kv_head_idx, prefix_len, q_len);
                }
                cur += 1;
                continue;
            }
            sl_head = get_head<HEAD_DIM>(sl, kv_head_idx);
            bool bands_ok = true;
            #pragma unroll
            for (int p = 0; p < N_PALETTE; ++p) {
                bands_ok = bands_ok && kvhead_k_ptr<HEAD_DIM>(sl_head, p) != 0
                                    && kvhead_v_ptr<HEAD_DIM>(sl_head, p) != 0;
            }
            if (!bands_ok) {
                if (tid == 0) {
                    printf("PPI8 GUARD: null band ptr in record at pos %d slice %d of "
                           "slot(b=%d,h=%d)\n",
                           cur, sl_idx, batch_idx, kv_head_idx);
                }
                cur += 1;
                continue;
            }
            int sl_off = (int)slice_offset(sl);
            int sl_len = (int)slice_len(sl);
            int remaining = sl_off + sl_len - in_blk0;
            tile_len = min(min(remaining, prefix_len - cur), I8_TILE_TOK);
            if (tile_len <= 0) { cur += 1; continue; } // defensive: skip hole
        } else {
            tile_len = min(kv_len - cur, I8_TILE_TOK);
        }
        // Round-robin tiles across shards — fresh tiles included (pinning
        // them to one shard measured as a ~9% SM-imbalance). The skip is
        // block-uniform (every thread computes identical cursor state),
        // so the staging barriers below stay convergent.
        bool mine = (tile_ord % num_splits) == split_idx;
        tile_ord += 1;
        if (!mine) {
            cur += tile_len;
            continue;
        }

        // -------------------- STAGE (all warps) --------------------
        if (!fresh) {
            constexpr int TOK_PER_WARP = I8_TILE_TOK / I8_WARPS; // 4
            const uint8_t* k_pal = kvhead_k_pal_map<HEAD_DIM>(sl_head);
            const uint8_t* v_pal = kvhead_v_pal_map<HEAD_DIM>(sl_head);

            // Raw-span fill: bulk-copy each palette's 32-token span into
            // smem with 16-byte cp.async. This is the coalescing fix for
            // the profiler's dominant finding — per-element extraction
            // straight from global wasted ~79% of its sectors; the bulk
            // copy is fully coalesced and the decodes below hit smem.
            //
            // Quant palettes copy their block run (≤ 36 B/dim). Dtype
            // palettes (F16/BF16/F8 — unsealed float prefixes, glue) copy
            // the raw element span; both sides' dtype spans cannot fit the
            // scratch simultaneously, so a tile containing ANY dtype
            // palette stages K and V SEQUENTIALLY through one
            // RAW_SLOT_D-strided region (two extra barriers). All-quant
            // tiles — the sealed production path — keep the simultaneous
            // K+V fill. Non-hop palettes (R16, F32, unaligned spans) keep
            // their global base and decode element-wise.
            auto issue_side = [&](int side, int slot_bytes, char* region) {
                #pragma unroll 1
                for (int p = 0; p < N_PALETTE; ++p) {
                    const char* gb = (const char*)(uintptr_t)(
                        side ? kvhead_v_ptr<HEAD_DIM>(sl_head, p)
                             : kvhead_k_ptr<HEAD_DIM>(sl_head, p));
                    const int fmt = side ? kvhead_v_fmt<HEAD_DIM>(sl_head, p)
                                         : kvhead_k_fmt<HEAD_DIM>(sl_head, p);
                    const int es = ArenaFormat::float_elem_size(fmt);
                    int bb = 0;
                    int span = 0;
                    if (es == 0) {
                        bb = ArenaAccessor::get_quant_block_bytes(fmt);
                        if (bb * SUB <= slot_bytes) span = SUB * bb;
                    } else if (I8_TILE_TOK * SUB * es <= slot_bytes) {
                        span = I8_TILE_TOK * SUB * es;
                    }
                    const char* eb = gb;
                    if (span > 0 && (((uintptr_t)gb & 15) == 0)) {
                        char* rb = region + p * slot_bytes;
                        // span is a multiple of 16 (SUB ≥ 16, even sizes).
                        for (int u = tid * 16; u < span; u += I8_THREADS * 16)
                            i8_cp_async16(rb + u, gb + u);
                        eb = rb;
                    }
                    if (tid == 0) {
                        s_ext_base[side][p] = eb;
                        s_ext_fmt[side][p] = fmt;
                        s_ext_bb[side][p] = (es == 0) ? bb : 0;
                        s_ext_scl[side][p] = side
                            ? kvhead_v_scale<HEAD_DIM>(sl_head, p)
                            : kvhead_k_scale<HEAD_DIM>(sl_head, p);
                    }
                }
            };
            bool tile_has_dtype = false;
            #pragma unroll 1
            for (int sp = 0; sp < 2 * N_PALETTE; ++sp) {
                const int p = sp & (N_PALETTE - 1);
                const int fmt = (sp >= N_PALETTE)
                    ? kvhead_v_fmt<HEAD_DIM>(sl_head, p)
                    : kvhead_k_fmt<HEAD_DIM>(sl_head, p);
                const int es = ArenaFormat::float_elem_size(fmt);
                tile_has_dtype |= (es == 1 || es == 2);
            }
            if (tile_has_dtype) {
                // Phase K only; V fills after the K extract reuses the region.
                issue_side(0, RAW_SLOT_D, s_raw_k);
            } else {
                issue_side(0, RAW_SLOT, s_raw_k);
                issue_side(1, RAW_SLOT, s_raw_v);
            }
            i8_cp_commit();
            i8_cp_wait0();

            // Palette tables for this slice (one per tile — slice-aligned
            // tiles cannot straddle maps). Consecutive slices usually share
            // routing, so the rebuild + its barrier are skipped when the
            // incoming maps match the cached ones. The comparison inputs are
            // identical for every thread, so the branch (and its barrier)
            // stay block-uniform.
            bool tbl_hit = (s_tbl_valid != 0) &&
                           pal_map_equal<HEAD_DIM>(k_pal, s_pal_cache) &&
                           pal_map_equal<HEAD_DIM>(v_pal, s_pal_cache + HEAD_DIM / 4);
            // Every thread must finish READING the cache before the rebuild
            // below WRITES it — a fast thread's cache update racing a slow
            // thread's comparison makes `tbl_hit` diverge, and a divergent
            // conditional barrier is arrival-counted: threads pair up across
            // DIFFERENT __syncthreads and silently release
            // (racecheck-confirmed; presented as a rare A/B flake). This
            // barrier also publishes the raw-span fill + s_ext metadata
            // (every thread drained its own cp.async groups above).
            __syncthreads();
            if (!tbl_hit) {
                for (int d = tid; d < HEAD_DIM; d += I8_THREADS) {
                    int p, rank;
                    prefill_pal_rank(k_pal, d, &p, &rank);
                    s_tbl_k[d] = (uint8_t)((p << 6) | rank);
                    prefill_pal_rank(v_pal, d, &p, &rank);
                    s_tbl_v[d] = (uint8_t)((p << 6) | rank);
                }
                for (int b = tid; b < HEAD_DIM / 4; b += I8_THREADS) {
                    s_pal_cache[b] = k_pal[b];
                    s_pal_cache[HEAD_DIM / 4 + b] = v_pal[b];
                }
                if (tid == 0) s_tbl_valid = 1;
                __syncthreads();
            }

            // K: each warp decodes its own tokens' dims straight from the
            // staged raw spans (or global for non-hop palettes), RoPEs,
            // and requants. No FP16 exchange hop — the rank→natural
            // permutation is in the table-indexed reads themselves, so K
            // staging is a single warp-private pass.
            for (int jj = 0; jj < TOK_PER_WARP; ++jj) {
                int j = warp * TOK_PER_WARP + jj;
                float x[N_WIN];
                if (j < tile_len) {
                    int within = in_blk0 + j;
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) {
                        int d = lane + 32 * w;
                        uint8_t t = s_tbl_k[d];
                        int p = (t >> 6) & (N_PALETTE - 1);
                        x[w] = i8_arena_elem(s_ext_fmt[0][p], s_ext_bb[0][p],
                                             s_ext_base[0][p], t & 63, within,
                                             s_ext_scl[0][p], SUB);
                    }
                    int pos = cur + j + (int)rope_base;
                    #pragma unroll
                    for (int w = 0; w < N_WIN / 2; ++w) {
                        float c, s;
                        i8_rope_cs<HEAD_DIM>(pos, lane + 32 * w, rope_cs, c, s);
                        float lo = x[w], hi = x[w + N_WIN / 2];
                        x[w] = lo * c - hi * s;
                        x[w + N_WIN / 2] = lo * s + hi * c;
                    }
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
                    s_k8[j][lane + 32 * w] = i8_quant(x[w], inv);
                    if (lane == 0) s_k_scale[j][w] = __float2half(scale);
                }
            }

            if (tile_has_dtype) {
                // Dtype tiles: every warp is done reading the K spans (each
                // warp's K extract completed above in program order, and the
                // barrier makes that global), so the region can host the V
                // spans. The second barrier publishes them + s_ext[1].
                __syncthreads();
                issue_side(1, RAW_SLOT_D, s_raw_k);
                i8_cp_commit();
                i8_cp_wait0();
                __syncthreads();
            }

            // V: read-through when every palette's format is an int8
            // passthrough family — arena bytes go straight to the V^T slab
            // in natural dim order, scale = block scale / palette scale.
            // Per-element extraction has no lane-width constraint, so this
            // engages at every head dim. Dead columns (j ≥ tile_len) stay
            // untouched: the compute phase masks them with P == 0.
            bool v_rt = true;
            #pragma unroll
            for (int p = 0; p < N_PALETTE; ++p)
                v_rt = v_rt && (s_ext_bb[1][p] > 0) &&
                       ArenaAccessor::is_int8_readthrough_format(s_ext_fmt[1][p]);
            if (v_rt) {
                // The warp's 4 tokens pack into ONE aligned 4-byte store
                // per dim (V8T_LD's byte columns are 4-way bank-conflicted
                // by construction — ldmatrix needs 16 | LD, conflict-free
                // byte columns need LD/4 coprime to 32 — so the lever is
                // 4× fewer stores). Dead trailing tokens pack as zero;
                // wholly-dead warps write nothing (masked by P == 0).
                const int j0 = warp * TOK_PER_WARP;
                if (j0 < tile_len) {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) {
                        int d = lane + 32 * w;
                        uint8_t t = s_tbl_v[d];
                        int p = (t >> 6) & (N_PALETTE - 1);
                        const char* blk =
                            s_ext_base[1][p] + (int64_t)(t & 63) * s_ext_bb[1][p];
                        int fmt = s_ext_fmt[1][p];
                        uint32_t pack = 0;
                        float sblk = 0.f;
                        #pragma unroll
                        for (int jj = 0; jj < TOK_PER_WARP; ++jj) {
                            if (j0 + jj < tile_len) {
                                Int8Sample smp = i8_rt_elem(fmt, blk, in_blk0 + j0 + jj);
                                pack |= (uint32_t)(uint8_t)smp.v << (8 * jj);
                                if (jj == 0) sblk = smp.s;
                            }
                        }
                        float inv = 1.f / s_ext_scl[1][p];
                        *(uint32_t*)&s_v8t[d][j0] = pack;
                        // Same-value cross-warp race: smp.s is per (dim,
                        // block) and the tile is one block, so every warp's
                        // write of s_v_scale[d] carries the identical value.
                        s_v_scale[d] = __float2half(sblk * inv);
                    }
                }
            } else {
                // FP fallback (asymmetric / curve / dtype V palettes): per
                // natural dim, two passes over the tile's tokens decoded
                // straight from the raw spans — max-abs, then requant. Two
                // passes instead of a 32-float register array, which is
                // guaranteed spill at the 64-reg 4-blocks/SM cap. No
                // barrier: the decode source is the (already published)
                // raw spans, not a cross-warp exchange slab.
                for (int rr = tid; rr < HEAD_DIM; rr += I8_THREADS) {
                    uint8_t t = s_tbl_v[rr];
                    int p = (t >> 6) & (N_PALETTE - 1), r = t & 63;
                    int fmt = s_ext_fmt[1][p], bb = s_ext_bb[1][p];
                    const char* pb = s_ext_base[1][p];
                    float scl = s_ext_scl[1][p];
                    float a = 0.f;
                    for (int j = 0; j < I8_TILE_TOK; ++j) {
                        float v = (j < tile_len)
                            ? i8_arena_elem(fmt, bb, pb, r, in_blk0 + j, scl, SUB)
                            : 0.f;
                        a = fmaxf(a, fabsf(v));
                    }
                    float scale = a / 127.f;
                    float inv = (scale > 0.f) ? 1.f / scale : 0.f;
                    for (int j4 = 0; j4 < I8_TILE_TOK; j4 += 4) {
                        uint32_t pack = 0;
                        #pragma unroll
                        for (int jj = 0; jj < 4; ++jj) {
                            int j = j4 + jj;
                            float v = (j < tile_len)
                                ? i8_arena_elem(fmt, bb, pb, r, in_blk0 + j, scl, SUB)
                                : 0.f;
                            pack |= (uint32_t)(uint8_t)i8_quant(v, inv) << (8 * jj);
                        }
                        *(uint32_t*)&s_v8t[rr][j4] = pack;
                    }
                    s_v_scale[rr] = __float2half(scale);
                }
            }
        } else {
            // Fresh tile: K/V straight from the packed inputs, natural order.
            for (int j = warp; j < I8_TILE_TOK; j += I8_WARPS) {
                int tok = (cur - prefix_len) + j; // fresh token index
                float x[N_WIN];
                float v[N_WIN];
                if (j < tile_len) {
                    const QT* kr = k_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
                    const QT* vr = v_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) {
                        x[w] = qt_to_f32<QT>(kr[lane + 32 * w]);
                        v[w] = qt_to_f32<QT>(vr[lane + 32 * w]);
                    }
                    int pos = cur + j + (int)rope_base;
                    #pragma unroll
                    for (int w = 0; w < N_WIN / 2; ++w) {
                        float c, s;
                        i8_rope_cs<HEAD_DIM>(pos, lane + 32 * w, rope_cs, c, s);
                        float lo = x[w], hi = x[w + N_WIN / 2];
                        x[w] = lo * c - hi * s;
                        x[w + N_WIN / 2] = lo * s + hi * c;
                    }
                } else {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; ++w) { x[w] = 0.f; v[w] = 0.f; }
                }
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
                    // stash V into the FP scratch for the per-dim pass below
                    s_fresh[j][lane + 32 * w] = __float2half(v[w]);
                }
            }
            __syncthreads();
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

        // Mask + scale. Column j's logical kv position is cur + j. The
        // causal horizons are per-tile transients (dead after this loop).
        int horizon[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int h = prefix_len + row_tok(i) + 1;
            if (h > kv_len) h = kv_len;
            horizon[i] = row_live(i) ? h : 0;
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int ja = s * 8 + n0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int j = ja + (i & 1);
                int row = i >> 1;
                bool ok = (j < tile_len) && (cur + j < horizon[row]);
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

        cur += tile_len;
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
            // Softmax state: both warps of a row-tile pair hold identical
            // m/l (duplicated QK); the dim_half==0 warp's lane 0 writes it.
            if (dim_half == 0 && (lane & 3) == 0) {
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
    cudaStream_t stream
) {
    int hpg = (n_kv_head > 0) ? n_head / n_kv_head : 1;
    if (hpg <= 0) hpg = 1;
    int block_m_tok = I8_M_ROWS / hpg;
    if (block_m_tok <= 0) block_m_tok = 1;
    uint32_t grid_x = (uint32_t)((max_q_len + block_m_tok - 1) / block_m_tok);
    if (grid_x == 0) grid_x = 1;

    // Split-KV factor: fan the tile walk across grid.z shards up to the
    // 4-blocks/SM residency limit. The short-q/long-prefix regime otherwise
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
    // residency limit (4 blocks/SM) but NEVER past it — floor, not ceil:
    // one block over the slot count starts a second wave and near-doubles
    // the makespan (measured: 320 blocks on 304 slots ran 1.69 → 2.36 ms).
    int num_splits = 1;
    if (base_blocks < s_sm_count) {
        num_splits = (4 * s_sm_count) / base_blocks;
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
        num_splits, partials);

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
