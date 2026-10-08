#pragma once
// INT8 dense entry, NARROW: the decode-width (M ≤ 8) KO dense matmul with K walked by the
// warps of one block, and no global partials.
//
// The split-K entry fills the card at decode width by cutting K across BLOCKS, which costs
// every block a global F32 partial per K tile, a fence and a counter atomic, and then a
// last block that re-reads all of K's partials through L2 in tile order while the rest of
// the card idles. At five rows that tail is a chain of L2 round trips the DRAM floor never
// pays for: the hyper-connection `down` (N = 416, K = 10,240) ran 10.4 µs for 4.5 MB.
//
// Here K is cut across the WARPS of a block instead, so the reduction never leaves shared
// memory. A block owns one 8-row output tile — one KO weight chunk per K tile, exactly what
// one warp of the 32-row tile owns elsewhere — and `blockDim.y` warps; warp `w` walks the
// contiguous range `[w·per, (w+1)·per)` of K tiles (`narrow_layout.cuh`) through its own
// cp.async ring of weight chunks, synchronised with `__syncwarp` alone. The activation
// rides in registers: lane (g, q) needs exactly the 32 bytes of token g's tile that pair
// with its weight fragment, eight 4-byte loads at stride 16, issued a tile ahead.
//
// One summation order. Warp 0 folds its own tiles into its running sum exactly as the
// unsplit kernel does (`ko_affine_fold` onto the accumulator). Every other warp folds each
// tile onto −0 — which yields the tile's fold itself, bit for bit, zero signs included —
// and stores it to its slot. After one barrier warp 0 adds the slots in tile order. That is
// `((0 + f₀) + f₁) + …`, the very chain of operations the unsplit kernel and the split-K
// reducer perform, so every output is the unsplit kernel's bits; the store is the shared
// `store_tile_output`, so every output width converts as it does there.
//
// Tokens: the m16n8k32 MMA's rows 0..7 carry the launch's ≤ 8 tokens; rows 8..15 are
// zero registers, and their outputs are never stored. MXFP4 never takes this path — its
// per-sub fold accumulates straight into the running sum, which a per-tile fold cannot
// reproduce.
//
// Segments: the same by-value `SplitKSegs` table as the split-K entry, its `tile_start`
// counted in 8-row tiles, so the stacked MoE trio (router | shared gate_up | gate) stays
// one launch.

#include "narrow_layout.cuh"

namespace grouped_tc {

// This lane's share of token `g`'s K tile `t`: the eight 4-byte words at byte offsets
// `q·4 + 16·j` of the tile (`j = 2·sub + half`), which are A-fragment registers a[0] and a[2]
// of sub `j / 2` — and the tile's (scale, Σ-field) half2. A token past the launch's rows is
// zeros, which nothing downstream stores.
__device__ __forceinline__ void load_narrow_act(
    const uint8_t* __restrict__ abytes, int g, int b_cnt, int k_tiles, int t, int q,
    uint32_t (&qa)[8], uint32_t& ds)
{
    if (g < b_cnt) {
        const int64_t flat = (int64_t)g * k_tiles + t;
        const uint8_t* qs = abytes + q8a1024_qs_off(flat) + q * 4;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qa[j] = __ldg(reinterpret_cast<const unsigned int*>(qs + j * 16));
        }
        ds = __ldg(reinterpret_cast<const unsigned int*>(abytes + q8a1024_ds_off(flat)));
    } else {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qa[j] = 0u;
        }
        ds = 0u;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, typename output_t>
static __device__ void quantized_matmul_dense_narrow_entry_int8(
    const SplitKSegs& segs,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst_all,
    int ncols_x, int total_batch, int dst_stride, int sum_norm)
{
    using block_c_t = block_compact_t<block_q_t>;
    static_assert(is_scale_separate<block_c_t>::value, "the narrow entry reads KO chunks");
    static_assert(!is_mxfp4_persub<block_c_t>::value,
                  "MXFP4's per-sub fold has no per-tile fold to sum; it never runs narrow");
    constexpr int CB = int8_chunk_bytes<block_c_t>::value;   // one 8-row chunk
    constexpr int AHEAD = narrow_ahead(CB);                  // chunks in flight per warp

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int warps = blockDim.y;
    const int g = lane >> 2;   // token (MMA row) and output pair this lane folds
    const int q = lane & 3;    // its k-quad; output rows 2q, 2q+1 of the tile
    const int b_cnt = total_batch;

    // This block's segment, by an unrolled compare over static indices — never `segs.w[s]`
    // at a runtime `s`, which copies the by-value table to local memory in every block (see
    // the split-K entry).
    const void* seg_w = segs.w[0];
    int seg_tile0 = segs.tile_start[0];
    int nrows_x = segs.n[0];
    int seg_col = segs.col_off[0];
    #pragma unroll
    for (int s = 1; s < SPLITK_MAX_SEGS; ++s) {
        if (s < segs.num && (int)blockIdx.y >= segs.tile_start[s]) {
            seg_w = segs.w[s];
            seg_tile0 = segs.tile_start[s];
            nrows_x = segs.n[s];
            seg_col = segs.col_off[s];
        }
    }
    const int row_base = ((int)blockIdx.y - seg_tile0) * 8;   // the tile's first row
    const auto* __restrict__ weights = reinterpret_cast<const block_c_t*>(seg_w);
    output_t* __restrict__ dst = dst_all + seg_col;

    const int k_tiles = ncols_x / K_TILE;
    const int per = narrow_tiles_per_warp(k_tiles, warps);
    const int k_lo = min(k_tiles, warp * per);
    const int k_hi = min(k_tiles, k_lo + per);

    extern __shared__ __align__(16) uint8_t narrow_dyn_smem[];
    uint8_t* ring = narrow_dyn_smem + warp * AHEAD * CB;
    float2* slots = reinterpret_cast<float2*>(narrow_dyn_smem + narrow_ring_bytes(CB, warps));

    // Launched with programmatic serialization (`pdl.cuh`): this warp's weight range into
    // L2 while the kernel before it finishes, then the wait, before the activation is read.
    for (int t = k_lo; t < k_hi; ++t) {
        pdl_prefetch_l2(&weights[(int64_t)t * (nrows_x / 8) + row_base / 8], CB, lane,
                        WARP_SIZE_TC);
    }
    pdl_launch_dependents();
    pdl_wait();

    // Prologue: the first AHEAD chunks, one cp.async group each — empty past the range, so
    // the `wait_group` count stays uniform.
    #pragma unroll
    for (int s = 0; s < AHEAD; ++s) {
        if (k_lo + s < k_hi) {
            load_warp_chunk_int8<block_c_t>(ring + s * CB, weights, k_lo + s, row_base,
                                            nrows_x, lane);
        }
        cp_async_commit();
    }

    const uint8_t* abytes = reinterpret_cast<const uint8_t*>(act);
    uint32_t qa[8];
    uint32_t ds = 0u;
    if (k_lo < k_hi) {
        load_narrow_act(abytes, g, b_cnt, k_tiles, k_lo, q, qa, ds);
    }
    // The operand's sum convention, as the two coefficients that rebuild Σx.
    const float sum_a = sum_norm ? 127.f : 0.f;
    const float sum_b = sum_norm ? 0.f : 1.f;
    const int rl = q * 2;
    // Warp 0's running sum for (token g; rows rl, rl+1): the unsplit kernel's accumulator.
    float run0 = 0.f;
    float run1 = 0.f;
    int ring_i = 0;

    for (int t = k_lo; t < k_hi; ++t) {
        // The next tile's activation, a tile ahead: its L2 round trip hides under this
        // tile's weight wait and MMA.
        uint32_t qn[8];
        uint32_t dsn = 0u;
        if (t + 1 < k_hi) {
            load_narrow_act(abytes, g, b_cnt, k_tiles, t + 1, q, qn, dsn);
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                qn[j] = 0u;
            }
        }

        cp_async_wait_group<AHEAD - 1>();   // this lane's copies of chunk t landed
        __syncwarp();                       // ... and every other lane's
        uint8_t* my_slot = ring + ring_i * CB;
        const block_c_t* blk = reinterpret_cast<const block_c_t*>(my_slot);
        // dm[rl] and dm[rl+1] are adjacent half2 (8 B, rl*4 is 8-aligned) → ONE int2 LDS.64.
        const int2 dd = *reinterpret_cast<const int2*>(&blk->dm[rl]);
        const float2 d0 = __half22float2(*reinterpret_cast<const half2*>(&dd.x));
        const float2 d1 = __half22float2(*reinterpret_cast<const half2*>(&dd.y));
        uint32_t b_frags[4][2];
        gemx_dequant_traits<block_c_t, half, half>::dequant_all_subs_int8(my_slot, lane, b_frags);
        __syncwarp();   // WAR: every lane has read the slot before it is refilled
        if (t + AHEAD < k_hi) {
            load_warp_chunk_int8<block_c_t>(my_slot, weights, t + AHEAD, row_base, nrows_x,
                                            lane);
        }
        cp_async_commit();
        ring_i = ring_i + 1 == AHEAD ? 0 : ring_i + 1;

        // The tile's exact int32 dot product: the same four k32 sub-MMAs, in the same two
        // accumulators, as the unsplit kernel — integer sums, so exact in any order.
        int32_t C0[4] = {0, 0, 0, 0};
        int32_t C1[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int sub = 0; sub < 4; sub += 2) {
            const uint32_t a0f[4] = {qa[2 * sub], 0u, qa[2 * sub + 1], 0u};
            const uint32_t a1f[4] = {qa[2 * sub + 2], 0u, qa[2 * sub + 3], 0u};
            fused_attn::mma_int8_m16n8k32(C0, a0f, b_frags[sub], C0);
            fused_attn::mma_int8_m16n8k32(C1, a1f, b_frags[sub + 1], C1);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) C0[i] += C1[i];

        const float2 a = __half22float2(*reinterpret_cast<const half2*>(&ds));
        // Warp 0 folds onto its running sum; the others onto −0, which gives the fold itself.
        float f[4];
        f[0] = warp == 0 ? run0 : -0.f;
        f[1] = warp == 0 ? run1 : -0.f;
        f[2] = -0.f;
        f[3] = -0.f;
        ko_affine_fold(f, C0, d0, d1, a, a, sum_a, sum_b);
        if (warp == 0) {
            run0 = f[0];
            run1 = f[1];
        } else {
            slots[(t - per) * WARP_SIZE_TC + lane] = make_float2(f[0], f[1]);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            qa[j] = qn[j];
        }
        ds = dsn;
    }
    cp_async_wait_group<0>();   // the trailing groups are empty; nothing is left in flight

    __syncthreads();   // every slotted fold is visible to warp 0
    if (warp == 0) {
        // The tiles past warp 0's range, in tile order. Eight slots' loads are issued before
        // their adds, so the chain pays one shared-memory round trip per eight tiles rather
        // than per tile.
        constexpr int BATCH = 8;
        const int n_slots = narrow_slot_tiles(k_tiles, warps);
        int i = 0;
        for (; i + BATCH <= n_slots; i += BATCH) {
            float2 v[BATCH];
            #pragma unroll
            for (int j = 0; j < BATCH; ++j) {
                v[j] = slots[(i + j) * WARP_SIZE_TC + lane];
            }
            #pragma unroll
            for (int j = 0; j < BATCH; ++j) {
                run0 += v[j].x;
                run1 += v[j].y;
            }
        }
        for (; i < n_slots; ++i) {
            const float2 v = slots[i * WARP_SIZE_TC + lane];
            run0 += v.x;
            run1 += v.y;
        }
        const float out[4] = {run0, run1, 0.f, 0.f};
        store_tile_output<output_t>(dst, out, dst_stride, row_base, 0, b_cnt, lane);
    }
}

} // namespace grouped_tc
