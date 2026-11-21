// The INT8 grouped matmul and the dense and split-K entries.
//
// `grouped_matmul_impl_int8`, the INT8 grouped body;
// `quantized_matmul_dense_entry_int8`, `quantized_matmul_dense_silu_entry_int8`
// and `quantized_matmul_dense_splitk_entry_int8`, the dense entries with a
// fused SiLU or split-K variant; and `splitk_reduce_tiles`, which reduces the
// split-K partials of one output tile.

// load_warp_chunk_int8 weight stage, then the int8 MMA + deferred fold per sub.
// N_SUB = m16 token sub-tiles per block. Mode-1 = 1 (Bm 16); mode-2 = larger Bm so each
// weight chunk's dequant is reused across N_SUB token sub-tiles (fewer weight re-reads).
//
// STAGES = K tiles the double-buffered (N_SUB ≤ 4) pipeline keeps resident: the one being
// computed plus STAGES−1 in flight, with a per-warp weight ring of STAGES−1 chunks and STAGES
// activation buffers. 2 is the ping-pong every unsplit launch runs — one tile in flight during
// the MMA, which is enough when a block walks the whole of K and the grid fills the card. A
// split-K block walks only a few tiles, and with one in flight it paid a full DRAM round
// trip per tile in series (ncu: the cp.async wait was ~45% of its stall samples); deeper stages
// keep more of a short run in flight. At STAGES = 2 this loop is exactly the ping-pong it
// generalises — a ring of one, buffers k & 1, `wait_group<0>`.
//
// TILE_PARTIALS (split-K only): store each K tile's fold to its own partial slot —
// `partial + k_blk · partial_stride` — and restart the accumulator from zero, instead of
// accumulating across tiles. For the affine KO formats every unsplit launch sums K as
// `((0 + f₀) + f₁) + …`, one tile's fold joining the accumulator in one add; the split-K
// reducer adds these per-tile partials in tile order from zero, which is that sum bit for
// bit. So a split launch is the unsplit kernel's bits whatever tiles each block takes, a
// row's output never depends on how many rows its wave carried, and the unsplit kernel
// carries nothing for it. (MXFP4's per-sub fold accumulates straight into the sum, which a
// per-tile partial cannot reproduce; it never splits — see `dense_qmatmul_with_plan`.)
//
// ACT_SILU: the activation is `silu` of F32 rows (`act_f32`, `act_f32_stride` apart) and
// each tile is quantized as it is loaded (`load_silu_f32_activations`) — no q8a128 operand
// exists; `act` is unread.
//
// RN = 8-row weight chunks per warp, so a block's row tile is N_TILE·RN rows (`row_tile_idx`
// counts tiles of that height). A warp dequants its RN chunks once per K tile and runs each A
// fragment it loads against all RN of them: at prefill width the per-K-tile shared-memory and
// L2 traffic per MMA falls with the tile area, which is what bounds the 32-row tile there (each
// m16 A fragment was loaded by every warp for a single n8 MMA, and every 32 output rows
// re-streamed the whole activation from L2). Per output, the tile order and the fold are the
// RN = 1 kernel's — each chunk keeps its own accumulator and folds through `ko_affine_fold` in
// tile order — so the bits do not depend on RN. A row tile may overhang N by whole warps
// (N is a multiple of 32, a warp covers 8·RN ≤ 32 rows): such a warp loads, computes and
// stores nothing, and still joins the activation loads and barriers.
template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB = 1,
          int STAGES = 2, bool TILE_PARTIALS = false, bool ACT_SILU = false, int RN = 1>
__device__ void grouped_matmul_impl_int8(
    const block_compact_t<block_q_t>* __restrict__ weights,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst,
    int ncols, int nrows, int y_stride, int dst_stride,
    int b_start, int b_cnt, int row_tile_idx,
    int8_t smem_A_i8[][N_SUB * 16][KI8_STRIDE],
    half2 smem_A_ds[][N_SUB * 16],
    uint8_t* smem_W_flat,
    // `SumScale::as_code()` for the activation operand: 0 raw, 1 Σx/amax.
    int sum_norm,
    // The K range this block reduces, in K_TILE blocks: `[k_lo, k_hi)`, the whole
    // of K when `k_hi < 0`. A split-K block covers one slice of it.
    int k_lo = 0,
    int k_hi = -1,
    // TILE_PARTIALS: the per-tile F32 partials, tile `k` at `partial + k · partial_stride`,
    // each laid out like `dst` (`dst_stride` wide). See the split-K dense entry.
    float* __restrict__ partial = nullptr,
    size_t partial_stride = 0,
    // ACT_SILU: the F32 rows the activation is `silu` of, and their stride in floats.
    const float* __restrict__ act_f32 = nullptr,
    int act_f32_stride = 0)
{
    using block_c_t = block_compact_t<block_q_t>;
    static_assert(32 % (8 * RN) == 0, "a warp's rows must divide the 32-row granule N comes in");
    // The operand's sum convention, as the two coefficients that rebuild Σx.
    // Uniform across the grid and resolved once here, so the k-loop below is
    // branchless and identical in shape for both conventions.
    const float sum_a = sum_norm ? 127.f : 0.f;
    const float sum_b = sum_norm ? 0.f : 1.f;
    const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
    const int warp_id = tid / WARP_SIZE_TC;
    const int lane = tid % WARP_SIZE_TC;
    const int row0 = row_tile_idx * N_TILE * RN;
    if (row0 >= nrows || b_cnt <= 0) return;

    const int warp_row_base = row0 + warp_id * 8 * RN;
    // Whether this warp's rows exist. Warp-uniform; always true for RN = 1, where a row tile
    // is the 32-row granule itself.
    const bool warp_live = RN == 1 || warp_row_base < nrows;
    const int tiles_per_row = ncols / K_TILE;  // block_q8a128 tiles per activation row

    // N_SUB m16 token sub-tiles × RN 8-row chunks per warp: frag_c[(t*RN + r)*4 .. +3] = tokens
    // [t*16, t*16+16) × rows [8r, 8r+8) of the warp's. The weight (and its dequant) is shared
    // across the sub-tiles — dequanted once per k_blk, reused.
    float frag_c[4 * N_SUB * RN] = {};
    const int k_begin = k_lo;
    const int k_blocks = k_hi < 0 ? ncols / K_TILE : k_hi;

    const int groupID  = lane >> 2;
    const int threadID = lane & 3;

    // Software-pipelined load. Per-warp weight slot (RING_I8=1, reused — the dequant drains it to
    // registers, freeing it for the next prefetch) + activation buffering by mode:
    //   N_SUB ≤ 4: DOUBLE-buffered (ping-pong) — tile k+1's weight AND activation prefetch in one
    //     cp.async group during tile k's MMA; the barrier waits only for warp skew, not load
    //     latency. (Single-buffering mode-2 was measured to regress 7-14% at M=4096: at its
    //     ~66% occupancy the intra-block overlap beats the occupancy the smem would buy back.)
    //   N_SUB = 8: SINGLE-buffered — the 128-token double buffer costs ~35 KB of smem and caps
    //     the kernel at 2 blocks/SM (ncu: 16.7% occupancy, nothing saturated — latency-bound).
    //     Halving it doubles resident blocks; the exposed activation reload (L2-served, ~85% hit)
    //     is covered by the extra block-level parallelism. Weight prefetch still overlaps the MMA
    //     (the slot is freed by the dequant); only the activation reload waits on a WAR barrier.
    constexpr int ABUF = (N_SUB >= 8) ? 1 : 2;
    static_assert(STAGES >= 2, "the pipeline keeps the computed tile plus at least one in flight");
    static_assert(ABUF == 2 || STAGES == 2, "the single-buffered N_SUB=8 path is not staged");
    // Tiles in flight ahead of the one being computed, and the weight ring that holds them: the
    // dequant drains a chunk to registers before the next prefetch, so a ring one short of the
    // stage count suffices.
    constexpr int AHEAD = (ABUF == 2) ? STAGES - 1 : 1;
    constexpr int WRING = AHEAD;
    constexpr int CB = int8_chunk_bytes<block_c_t>::value;   // one 8-row chunk
    constexpr int SLOT = RN * CB;                             // a warp's chunks of one K tile
    uint8_t* warp_ring = smem_W_flat + warp_id * WRING * SLOT;
    // Tile `t`'s weight slot and activation buffer, counted from this block's first tile.
    auto w_slot = [&](int t) { return warp_ring + ((t - k_begin) % WRING) * SLOT; };
    auto a_buf = [&](int t) { return (ABUF == 1) ? 0 : (t - k_begin) % STAGES; };

    // Prologue: the first AHEAD tiles — weight chunk + activation tile, one cp.async group each.
    // Deeper pipelines commit an (empty) group past the end too, so `wait_group` counts stay
    // uniform; the ping-pong commits only what it loads, as it always has.
    #pragma unroll
    for (int s = 0; s < AHEAD; ++s) {
        const int t = k_begin + s;
        if (t < k_blocks) {
            if (warp_live) {
                load_warp_chunk_int8<block_c_t, RN>(w_slot(t), weights, t, warp_row_base, nrows, lane);
            }
            load_act_tile<N_SUB, ACT_SILU>(act, act_f32, act_f32_stride, b_start, b_cnt, t,
                                           tiles_per_row, tid, sum_norm,
                                           smem_A_i8[a_buf(t)], smem_A_ds[a_buf(t)]);
            if constexpr (STAGES == 2) cp_async_commit();
        }
        if constexpr (STAGES > 2) cp_async_commit();
    }

    for (int k_blk = k_begin; k_blk < k_blocks; ++k_blk) {
        const int ab = a_buf(k_blk);     // this tile's activation buffer
        uint8_t* my_slot = w_slot(k_blk);
        cp_async_wait_group<AHEAD - 1>(); // tile k_blk (weight + activation) resident

        // Inline scales + up-front dequant: read this warp's chunks into registers. Intra-warp
        // (no barrier needed); frees the weight slot for the next prefetch. This thread owns
        // fold rows (threadID*2, threadID*2+1) of each chunk. Affine formats read the per-128
        // (scale,min) pair from blk.dm[row]; MXFP4 reads the four per-sub E8M0 scales per fold
        // row instead (its per-sub fold below never touches dm).
        const int rl = threadID * 2;
        float2 d0[RN];                     // (d, m) row rl   — affine fold only
        float2 d1[RN];                     // (d, m) row rl+1 — affine fold only
        float f0[RN][4], f1[RN][4];        // per-sub scales  — mxfp4 fold only
        // Lane-major dequant: this lane's 4 subs are stored contiguously, so each stream is
        // pulled in ONE wide LDS (ql int4, plus crumb/hi for Q5/Q6, or 2 int4 for Q8) instead
        // of 4 per-sub loads — fewer MIO instructions, still bank-conflict-free.
        uint32_t b_frags[RN][4][2];
        if (warp_live) {
            #pragma unroll
            for (int r = 0; r < RN; ++r) {
                const uint8_t* chunk = my_slot + r * CB;
                const block_c_t* blk = reinterpret_cast<const block_c_t*>(chunk);
                d0[r] = make_float2(0.f, 0.f);
                d1[r] = make_float2(0.f, 0.f);
                if constexpr (is_mxfp4_persub<block_c_t>::value) {
                    gemx_dequant_traits<block_c_t, half, half>::load_sub_scales(chunk, rl, f0[r], f1[r]);
                } else {
                    // dm[rl] and dm[rl+1] are adjacent half2 (8 B, rl*4 is 8-aligned) → ONE
                    // int2 LDS.64.
                    const int2 dd = *reinterpret_cast<const int2*>(&blk->dm[rl]);
                    d0[r] = __half22float2(*reinterpret_cast<const half2*>(&dd.x));
                    d1[r] = __half22float2(*reinterpret_cast<const half2*>(&dd.y));
                    f0[r][0] = 0.f; f0[r][1] = 0.f; f0[r][2] = 0.f; f0[r][3] = 0.f;
                    f1[r][0] = 0.f; f1[r][1] = 0.f; f1[r][2] = 0.f; f1[r][3] = 0.f;
                }
                gemx_dequant_traits<block_c_t, half, half>::dequant_all_subs_int8(chunk, lane, b_frags[r]);
            }
        }

        __syncthreads();  // RAW: tile k activation visible to all warps; also guarantees tile
                          // k-1's MMA (which read the buffer the prefetch below fills) finished.

        if constexpr (ABUF == 2) {
            // Prefetch tile k+AHEAD: its weight chunks land in the slot the dequant above just
            // drained (the ring is AHEAD long, so tile k+AHEAD maps to tile k's slot) and its
            // activation in the buffer tile k-1's MMA released, ONE cp.async group. Both overlap
            // the MMA below, which reads only registers and buffer `ab`.
            const int next = k_blk + AHEAD;
            if (next < k_blocks) {
                if (warp_live) {
                    load_warp_chunk_int8<block_c_t, RN>(w_slot(next), weights, next, warp_row_base, nrows, lane);
                }
                load_act_tile<N_SUB, ACT_SILU>(act, act_f32, act_f32_stride, b_start, b_cnt,
                                               next, tiles_per_row, tid, sum_norm,
                                               smem_A_i8[a_buf(next)], smem_A_ds[a_buf(next)]);
                if constexpr (STAGES == 2) cp_async_commit();
            }
            if constexpr (STAGES > 2) cp_async_commit();
        } else {
            // Single buffer: only the weight prefetch overlaps the MMA (its slot is already
            // free). The activation reload waits for the WAR barrier after the sub-tile loop.
            if (k_blk + 1 < k_blocks) {
                if (warp_live) {
                    load_warp_chunk_int8<block_c_t, RN>(my_slot, weights, k_blk + 1, warp_row_base, nrows, lane);
                }
                cp_async_commit();
            }
        }

        // Per token sub-tile. The dequanted weight (b_frags + scales) is REUSED across all
        // N_SUB sub-tiles — that's the mode-2 win (one weight dequant amortized over
        // N_SUB·16 tokens) — and each A fragment across the warp's RN chunks. Each sub-tile t
        // has its own activation (smem at t*16) and outputs frag_c[(t*RN + r)*4..].
        //
        // A sub-tile holding no live token is skipped: its rows are the loader's
        // zero fill and nothing below ever stores them, so the result is the
        // same bits without its ldmatrix and MMA work. `b_cnt` is block-uniform,
        // so the guard costs no divergence, and `t` stays a compile-time index.
        // A wide prefill tile (Bm 128 over ~80 rows an expert) is a third
        // padding.
        if (warp_live) {
            #pragma unroll
            for (int t = 0; t < N_SUB; ++t) {
                if (t * 16 >= b_cnt) continue;
                const float2 a0 = __half22float2(smem_A_ds[ab][t * 16 + groupID]);      // token-half A
                const float2 a1 = __half22float2(smem_A_ds[ab][t * 16 + groupID + 8]);  // token-half B
                if constexpr (is_mxfp4_persub<block_c_t>::value) {
                    // PER-SUB fold: each 32-K sub's exact int32 sum scaled by its own
                    // E8M0 `2^(e_sub-128)` (the activation scale is per-128, so it is
                    // constant across the four subs). One reusable accumulator per chunk —
                    // the fold drains it straight after its MMA. MXFP4 is centred (m = 0): no
                    // activation-sum term.
                    #pragma unroll
                    for (int sub = 0; sub < 4; ++sub) {
                        uint32_t af[4];
                        fused_attn::load_a_frag_m16k32_ldmatrix(af, &smem_A_i8[ab][t * 16][sub * 32], KI8_STRIDE, lane);
                        #pragma unroll
                        for (int r = 0; r < RN; ++r) {
                            float* acc = &frag_c[(t * RN + r) * 4];
                            int32_t C[4] = {0, 0, 0, 0};
                            fused_attn::mma_int8_m16n8k32(C, af, b_frags[r][sub], C);
                            acc[0] += (f0[r][sub] * a0.x) * (float)C[0];
                            acc[1] += (f1[r][sub] * a0.x) * (float)C[1];
                            acc[2] += (f0[r][sub] * a1.x) * (float)C[2];
                            acc[3] += (f1[r][sub] * a1.x) * (float)C[3];
                        }
                    }
                } else {
                    // Affine per-128 fold: the tile's four MMAs over paired K segments
                    // (`kseg_lo`) sum into one int32 per chunk, and the (d, m) pair folds once
                    // over it. Consecutive MMAs go to different chunks, so with RN ≥ 2 they are
                    // independent; a lone chunk splits its MMAs over two accumulators instead.
                    constexpr int NACC = RN == 1 ? 2 : 1;
                    int32_t C[RN][NACC][4];
                    #pragma unroll
                    for (int r = 0; r < RN; ++r) {
                        #pragma unroll
                        for (int j = 0; j < NACC; ++j) {
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) C[r][j][i] = 0;
                        }
                    }
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        uint32_t af[4];
                        load_a_frag_kseg(af, &smem_A_i8[ab][t * 16][0], kseg_lo(q), kseg_lo(q) + 32,
                                         lane);
                        const int s = (q & 1) * 2;   // the pair's first sub
                        const int h = q >> 1;        // which half of each sub
                        #pragma unroll
                        for (int r = 0; r < RN; ++r) {
                            const uint32_t bq[2] = {b_frags[r][s][h], b_frags[r][s + 1][h]};
                            fused_attn::mma_int8_m16n8k32(C[r][q % NACC], af, bq, C[r][q % NACC]);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < RN; ++r) {
                        if constexpr (NACC == 2) {
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) C[r][0][i] += C[r][1][i];
                        }
                        // `sum_a`/`sum_b` are hoisted from the operand's `SumScale` before the
                        // k-loop, so the Σx rebuild inside is one FMA + one multiply either way
                        // — no branch in the inner loop, and no second kernel.
                        ko_affine_fold(&frag_c[(t * RN + r) * 4], C[r][0], d0[r], d1[r], a0, a1,
                                       sum_a, sum_b);
                    }
                }
            }
        }

        if constexpr (TILE_PARTIALS) {
            // This tile's fold to its own slot, and the accumulator restarts at zero.
            float* slot = partial + (size_t)k_blk * partial_stride;
            if (warp_live) {
                #pragma unroll
                for (int t = 0; t < N_SUB; ++t) {
                    const int rem = b_cnt - t * 16;
                    const int cnt = rem < 0 ? 0 : (rem > 16 ? 16 : rem);
                    #pragma unroll
                    for (int r = 0; r < RN; ++r) {
                        store_tile_output<float>(slot, &frag_c[(t * RN + r) * 4], dst_stride,
                                                 warp_row_base + 8 * r, b_start + t * 16, cnt, lane);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 4 * N_SUB * RN; ++i) frag_c[i] = 0.f;
        }

        if constexpr (ABUF == 1) {
            // WAR: every warp is done reading tile k's activation — safe to reload in place.
            __syncthreads();
            if (k_blk + 1 < k_blocks) {
                load_act_tile<N_SUB, ACT_SILU>(act, act_f32, act_f32_stride, b_start, b_cnt,
                                               k_blk + 1, tiles_per_row, tid, sum_norm,
                                               smem_A_i8[0], smem_A_ds[0]);
                cp_async_commit();
            }
        }
    }

    // One store per (sub-tile, chunk): tile t → tokens [b_start+t*16, +16) of chunk r's rows.
    // store_tile_output clamps via the token count, so partial/empty tiles write nothing. A
    // split-K block has stored its per-tile partials already; the entry finishes the reduction.
    if constexpr (!TILE_PARTIALS) {
        if (warp_live) {
            #pragma unroll
            for (int t = 0; t < N_SUB; ++t) {
                const int rem = b_cnt - t * 16;
                const int cnt = rem < 0 ? 0 : (rem > 16 ? 16 : rem);
                #pragma unroll
                for (int r = 0; r < RN; ++r) {
                    store_tile_output<output_t>(dst, &frag_c[(t * RN + r) * 4], dst_stride,
                                                warp_row_base + 8 * r, b_start + t * 16, cnt, lane);
                }
            }
        }
    }
}

// INT8 dense entry (regular non-MoE QMatMul): one weight, implicit tile schedule —
// blockIdx.x → the 16·N_SUB-token batch slice, blockIdx.y → the 32·RN-row tile. Launched
// from run_quantized_matmul on ytype==3. Each row folds K tile by tile in order, the
// same chain whatever the batch width or tile shape, and the split-K reducer reproduces it
// bit for bit.
template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB = 1, int RN = 1>
static __device__ void quantized_matmul_dense_entry_int8(
    const block_compact_t<block_q_t>* __restrict__ weights,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int total_batch, int y_stride, int dst_stride,
    // The activation operand's `SumScale::as_code()`, passed straight through to
    // the impl, which turns it into two hoisted coefficients.
    int sum_norm)
{
    constexpr int BATCH = N_SUB * 16;   // tokens per block (mode-1: 16, mode-2: 16·N_SUB)
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    const int row_tile_idx = blockIdx.y;

    __shared__ __align__(16) int8_t smem_A_i8[2][BATCH][KI8_STRIDE];   // double-buffered
    __shared__ __align__(16) half2 smem_A_ds[2][BATCH];
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * RN * RING_I8 * int8_chunk_bytes<block_compact_t<block_q_t>>::value];

    // The int8 impl is KO-only (inline-scale k1024 chunks). Only instantiate it for KO; for
    // any non-KO type the call is discarded so the kernel is a no-op (never dispatched to).
    if constexpr (is_scale_separate<block_compact_t<block_q_t>>::value) {
        // Launched with programmatic serialization (`pdl.cuh`): the row tile's first
        // K tiles into L2 while the kernel before it finishes, then the wait, before the
        // activation is read. Only the first tiles: a wide grid's later blocks start
        // after the predecessor has finished anyway, and a block's pipeline covers the
        // rest of its K once it is running.
        {
            constexpr int CB = int8_chunk_bytes<block_compact_t<block_q_t>>::value;
            constexpr int PREFETCH_TILES = 4;
            const int row0 = row_tile_idx * N_TILE * RN;
            const int t_id = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
            const int k_tiles = ncols_x / K_TILE;
            if (row0 < nrows_x) {
                // The tile's rows that exist: the last row tile may overhang N.
                const int rows = min(N_TILE * RN, nrows_x - row0);
                for (int t = 0; t < k_tiles && t < PREFETCH_TILES; ++t) {
                    pdl_prefetch_l2(&weights[(int64_t)t * (nrows_x / 8) + row0 / 8],
                                    (rows / 8) * CB, t_id, 4 * WARP_SIZE_TC);
                }
            }
            pdl_launch_dependents();
            pdl_wait();
        }
        grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB, 2, false, false, RN>(
            weights, act, dst, ncols_x, nrows_x, y_stride, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat, sum_norm);
    }
}

// INT8 dense entry over a FUSED activation: `dst = silu(proj[:, 0..K]) · Wᵀ`, with the
// activation quantized tile by tile as the impl loads it (ACT_SILU) — the matmul the
// hyper-connection's `up` projection runs, without the `gr_silu_q8` launch that wrote its
// operand. `proj` is the bottleneck's `[M, proj_stride]` F32 output, read through its row
// stride. Same tile schedule as the dense entry, and the same bits as `gr_silu_q8` followed
// by it.
//
// Pipeline depth (see `grouped_matmul_impl_int8`'s STAGES): the computed tile plus three in
// flight in mode 1. The `up` contraction is the hyper-connection bottleneck, K = 384 — three
// K tiles — so a block's whole K is issued in its prologue instead of one DRAM round trip per
// tile; mode 2 keeps the ping-pong its prefill widths are tuned on.
template <int N_SUB>
constexpr int dense_silu_stages() { return N_SUB == 1 ? 4 : 2; }

template <int qk, int qi, typename block_q_t, int vdr, typename output_t, int N_SUB = 1>
static __device__ void quantized_matmul_dense_silu_entry_int8(
    const block_compact_t<block_q_t>* __restrict__ weights,
    const float* __restrict__ proj,
    int proj_stride,
    output_t* __restrict__ dst,
    int ncols_x, int nrows_x, int total_batch, int dst_stride,
    int sum_norm)
{
    constexpr int BATCH = N_SUB * 16;
    constexpr int STAGES = dense_silu_stages<N_SUB>();
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    const int row_tile_idx = blockIdx.y;

    __shared__ __align__(16) int8_t smem_A_i8[STAGES][BATCH][KI8_STRIDE];
    __shared__ __align__(16) half2 smem_A_ds[STAGES][BATCH];
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * (STAGES - 1) * int8_chunk_bytes<block_compact_t<block_q_t>>::value];

    if constexpr (is_scale_separate<block_compact_t<block_q_t>>::value) {
        // Launched with programmatic serialization (`pdl.cuh`): the row tile's whole
        // weight — `N_TILE / 8` chunks per K tile — into L2 while the kernel before it
        // finishes, then the wait, before `proj` is read.
        {
            constexpr int CB = int8_chunk_bytes<block_compact_t<block_q_t>>::value;
            const int row0 = row_tile_idx * N_TILE;
            const int t_id = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
            if (row0 < nrows_x) {
                for (int t = 0; t < ncols_x / K_TILE; ++t) {
                    pdl_prefetch_l2(&weights[(int64_t)t * (nrows_x / 8) + row0 / 8],
                                    (N_TILE / 8) * CB, t_id, 4 * WARP_SIZE_TC);
                }
            }
            pdl_launch_dependents();
            pdl_wait();
        }
        grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, N_SUB, STAGES, false, true>(
            weights, nullptr, dst, ncols_x, nrows_x, 0, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat, sum_norm,
            0, -1, nullptr, 0, proj, proj_stride);
    }
}

// INT8 dense entry, SPLIT-K: the same tiles, K cut into `gridDim.z` slices of
// ceil(K-tiles / gridDim.z) tiles each. The host picks the depth (q8a128_dense_k_splits)
// so that no slice is empty; the bits do not depend on it (see TILE_PARTIALS).
//
// For a decode-width projection with a narrow N — the hyper-connection `down`
// (N = 416, K = 10,240) — the unsplit grid is ceil(M/16) × N/32 = 13 blocks on a
// 110-SM card, each walking 80 K tiles with one chunk in flight: ncu measured
// 8% achieved occupancy and 7% of DRAM. Cutting K gives every slice its own
// block, so the grid fills the card and the bytes in flight scale with it.
//
// Each block stores one F32 partial per K tile it covers to `ws[k]` (`[M, N]` each),
// then counts in on `counters[tile]`. The block that counts in LAST reduces the tile: it
// reads every tile's partial — its own included, from `ws`, not registers — in tile
// order and stores the result. That is the chain every unsplit dense launch folds in too
// (`((0 + f₀) + f₁) + …`), so the result is the unsplit kernel's bit for bit, whichever
// block arrives last, with no float atomics. The last block also
// returns its counter to zero, which is what lets the counters live in a buffer
// zeroed once and reused by every launch on the stream.
//
// Pipeline depth of a split-K block (see `grouped_matmul_impl_int8`'s STAGES): the computed
// tile plus two in flight — a short slice is issued whole, a longer one streams.
constexpr int SPLITK_STAGES = 3;

// The split-K reducer's threads (one block of the dense tile) and the partials each thread
// keeps in flight per L2 round trip, whatever its share of the outputs.
constexpr int SPLITK_RED_THREADS = 4 * WARP_SIZE_TC;
constexpr int SPLITK_RED_INFLIGHT = 32;

// The last block's reduction of one output tile: thread `tid` owns outputs `tid + j·THREADS`
// for j < OUTS (those below `outs`), and sums each over all `k_blocks` per-tile partials in
// tile order from zero, `BATCH_K` tiles per round trip — every load of a round trip is issued
// before its adds, and the adds run in tile order, so the chain is the unsplit kernel's.
// `__ldcg` reads past L1, which the other blocks' stores never touched. N is a multiple of 32
// (the launcher refuses anything else), so every output row of the tile exists.
template <int OUTS, int BATCH_K, typename output_t>
static __device__ __forceinline__ void splitk_reduce_tiles(
    const float* __restrict__ ws, size_t slice, int k_blocks,
    output_t* __restrict__ dst, int dst_stride, int b_start, int row0, int outs, int tid)
{
    // Offsets within one tile's `[M, N]` partial — the scratch holds 2²² floats, so 32 bits
    // index it.
    unsigned int off[OUTS];
    bool live[OUTS];
    float acc[OUTS];
    #pragma unroll
    for (int j = 0; j < OUTS; ++j) {
        const int o = tid + j * SPLITK_RED_THREADS;
        live[j] = o < outs;
        off[j] = (unsigned int)((b_start + o / N_TILE) * dst_stride + row0 + o % N_TILE);
        acc[j] = 0.f;
    }
    for (int s = 0; s < k_blocks; s += BATCH_K) {
        float v[OUTS][BATCH_K];
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            #pragma unroll
            for (int i = 0; i < BATCH_K; ++i) {
                v[j][i] = live[j] && s + i < k_blocks
                              ? __ldcg(&ws[(size_t)(s + i) * slice + off[j]]) : 0.f;
            }
        }
        #pragma unroll
        for (int j = 0; j < OUTS; ++j) {
            #pragma unroll
            for (int i = 0; i < BATCH_K; ++i) {
                if (s + i < k_blocks) acc[j] += v[j][i];
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < OUTS; ++j) {
        if (live[j]) {
            if constexpr (std::is_same_v<output_t, float>) dst[off[j]] = acc[j];
            else if constexpr (std::is_same_v<output_t, half>) dst[off[j]] = __float2half_rn(acc[j]);
            else if constexpr (std::is_same_v<output_t, __nv_bfloat16>) dst[off[j]] = __float2bfloat16_rn(acc[j]);
        }
    }
}

// `segs` — the weight segments of the shared output row (`splitk_segs.cuh`).
template <int qk, int qi, typename block_q_t, int vdr, typename output_t>
static __device__ void quantized_matmul_dense_splitk_entry_int8(
    const SplitKSegs& segs,
    const block_q8a128* __restrict__ act,
    output_t* __restrict__ dst_all,
    int ncols_x, int total_batch, int dst_stride, int sum_norm,
    float* __restrict__ ws_all,
    unsigned int* __restrict__ counters)
{
    constexpr int BATCH = 16;
    const int b_start = blockIdx.x * BATCH;
    const int b_cnt = min(BATCH, total_batch - b_start);
    // This block's segment: the tiles are contiguous per segment, in order. Selected by
    // an unrolled compare over static indices, never `segs.w[seg]`: a runtime index into
    // a by-value parameter struct makes the compiler copy the whole table to local
    // memory in every block, and that alone cost a single-weight launch 15–30%.
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
    const int row_tile_idx = (int)blockIdx.y - seg_tile0;
    const auto* __restrict__ weights = reinterpret_cast<const block_compact_t<block_q_t>*>(seg_w);
    // The segment's columns of the shared output and of every K tile's partial.
    output_t* __restrict__ dst = dst_all + seg_col;
    float* __restrict__ ws = ws_all + seg_col;
    const int splits = gridDim.z;
    const int k_blocks = ncols_x / K_TILE;
    // This block's slice.
    const int slice_tiles = (k_blocks + splits - 1) / splits;
    const int k_lo = min(k_blocks, (int)blockIdx.z * slice_tiles);
    const int k_hi = min(k_blocks, k_lo + slice_tiles);

    constexpr int STAGES = SPLITK_STAGES;
    __shared__ __align__(16) int8_t smem_A_i8[STAGES][BATCH][KI8_STRIDE];
    __shared__ __align__(16) half2 smem_A_ds[STAGES][BATCH];
    __shared__ uint8_t smem_W_flat[(N_TILE / 8) * (STAGES - 1) * int8_chunk_bytes<block_compact_t<block_q_t>>::value];
    __shared__ bool s_last;

    if constexpr (is_scale_separate<block_compact_t<block_q_t>>::value) {
        // Launched with programmatic serialization (`pdl.cuh`): this slice's weight
        // span — `N_TILE / 8` consecutive chunks per K tile — into L2 while the kernel
        // before it finishes, then the wait, before the activation, the partials or
        // the counters are touched.
        {
            constexpr int CB = int8_chunk_bytes<block_compact_t<block_q_t>>::value;
            const int row0 = row_tile_idx * N_TILE;
            const int t_id = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
            if (row0 < nrows_x) {
                for (int t = k_lo; t < k_hi; ++t) {
                    pdl_prefetch_l2(&weights[(int64_t)t * (nrows_x / 8) + row0 / 8],
                                    (N_TILE / 8) * CB, t_id, 4 * WARP_SIZE_TC);
                }
            }
            pdl_launch_dependents();
            pdl_wait();
        }
        const size_t slice = (size_t)total_batch * dst_stride;
        grouped_matmul_impl_int8<qk, qi, block_q_t, vdr, output_t, 1, STAGES, true>(
            weights, act, dst, ncols_x, nrows_x, 0, dst_stride,
            b_start, b_cnt, row_tile_idx, smem_A_i8, smem_A_ds, smem_W_flat, sum_norm,
            k_lo, k_hi, ws, slice);

        // Publish this slice's partial before counting in, so the last block reads it.
        __threadfence();
        __syncthreads();
        const int tid = threadIdx.y * WARP_SIZE_TC + threadIdx.x;
        const int tile = blockIdx.y * gridDim.x + blockIdx.x;
        if (tid == 0) {
            s_last = atomicAdd(&counters[tile], 1u) == (unsigned int)(splits - 1);
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        __threadfence();

        // The reduction: 32 output rows × b_cnt tokens, each output summed over every K
        // tile's partial in tile order from zero — the unsplit chain, so every bit of the
        // result is the unsplit kernel's. It runs in one block per output tile while the
        // rest of the card idles, so it is bound by L2 round trips. The block's 128 threads
        // own ceil(outs / 128) outputs each and hold a fixed budget of 32 partials in flight
        // per round trip, shared out across those outputs: one output per thread (up to
        // four tokens — every draft and single-row decode) walks 32 tiles a trip, two (a
        // five-row verify) 16, four (a full token tile) 8. The hyper-connection `down` has
        // 80 K tiles, so a five-row verify reduces in 5 round trips and a draft row in 3,
        // where a fixed 8-tile batch paid 10 at every width.
        const int row0 = row_tile_idx * N_TILE;
        const int outs = b_cnt * N_TILE;
        if (outs <= SPLITK_RED_THREADS) {
            splitk_reduce_tiles<1, SPLITK_RED_INFLIGHT>(ws, slice, k_blocks, dst, dst_stride,
                                                        b_start, row0, outs, tid);
        } else if (outs <= 2 * SPLITK_RED_THREADS) {
            splitk_reduce_tiles<2, SPLITK_RED_INFLIGHT / 2>(ws, slice, k_blocks, dst, dst_stride,
                                                            b_start, row0, outs, tid);
        } else {
            splitk_reduce_tiles<4, SPLITK_RED_INFLIGHT / 4>(ws, slice, k_blocks, dst, dst_stride,
                                                            b_start, row0, outs, tid);
        }
        if (tid == 0) {
            counters[tile] = 0u;
        }
    }
}
