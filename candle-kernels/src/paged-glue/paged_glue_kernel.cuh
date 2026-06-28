#pragma once
// =============================================================================
// paged_glue_kernel.cuh — reprojection "glue" attention (decode-derivative).
//
// The glue forward attends G query tokens (x hpg heads) over a long QUANTIZED
// prefix, writes the glue K/V into the writer chunks, and lets glue attend
// earlier glue. It is the decode workload (few queries, long quantized prefix)
// with G>1 query tokens, so — unlike the paged-prefill kernel, which
// re-dequantizes the prefix once per Q-tile — this kernel streams each prefix
// column's K/V exactly ONCE and reuses it across all G x hpg query rows
// (dequant-once). The G x hpg flash-state lives in shared memory; the warps
// cooperate over the streamed columns rather than each owning a per-warp partial
// (which would not scale past hpg rows). See docs/paged_glue_kernel.md.
//
// v0: one block per (slot, kv_head); a per-column manual dot (correctness
// first); query tokens tiled by G_TILE so the smem flash-state fits. The INT8
// m16n8k32 MMA + split-KV + combine are layered on after the bit-exact gate.
//
// FORWARD WINDOW (fwd_window / b_avail): glue rows attend the prefix (A) and
// earlier glue causally AND, additionally, the first `fwd_b` columns of the next
// section (B), so glue is generated as a bridge INTO B, not a blind continuation
// of A. B columns are read-only here (never scattered — B is immutable) and must
// already be resident in the slot's position_map at [kv_len, kv_len+fwd_b) with
// assembled positions in col_actual_pos. The window is asymmetric BY DESIGN:
// backward is unbounded (A is true, fixed context — free and consistent to read),
// forward is capped (B's leading keys are stale, written against B's original,
// now-deleted neighbourhood — read only as far as B's heads reach back, ~16-32
// tokens). fwd_b==0 is bit-identical to the backward-only kernel.
// =============================================================================

#include "../paged-decode/int8_decode_kernel.cuh"

namespace paged_glue {

// Query tokens processed per block-pass; larger glue runs fan out across
// gridDim.z (those tiles overlap on the SMs). The Q rows + O accumulators for
// G_TILE x hpg rows live in dynamic shared memory: 2*(G_TILE*8)*128*4 bytes,
// i.e. ~16 KB per unit of G_TILE at HEAD_DIM=128. This kernel is
// OCCUPANCY-bound, not dequant-bound (the dequant load is ~20% of the runtime;
// the flash-state smem caps blocks/SM). G_TILE=2 -> ~24 KB total smem -> 3-4
// blocks/SM, which measured ~4.3x faster than the smem-heavy G_TILE=8 (1 block/
// SM). Lower G_TILE trades more gridDim.z blocks for higher occupancy.
constexpr int GLUE_G_TILE = 2;

// One block per (slot, kv_head). All warps cooperate: each streamed prefix /
// glue column is dequantized once into smem and scored against every resident
// glue row.
template <typename Q_T, typename T, typename O, int HEAD_DIM, int WARPS_PER_BLOCK>
__global__ void paged_glue_kernel(
    const Q_T* __restrict__ q,
    const uint8_t* __restrict__ headers_ptr,
    O* __restrict__ out,
    int batch,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,
    const T* __restrict__ v_new,
    const float* __restrict__ rope_cs,
    bool rope_interleaved,
    const uint32_t* __restrict__ cu_seqlens_q,
    const uint32_t* __restrict__ q_lens,
    const uint32_t* __restrict__ kv_lens,
    const uint32_t* __restrict__ col_actual_pos,
    const uint32_t* __restrict__ cu_kvlens,
    const uint32_t* __restrict__ glue_write_slice,
    const uint32_t* __restrict__ glue_write_in_blk,
    uint32_t fwd_window,                       // forward B-head window (tokens); 0 == backward-only
    const uint32_t* __restrict__ b_avail       // per-slot count of B columns resident after kv_len
) {
    constexpr int N_PALETTE = 4;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int VEC = HEAD_DIM / 32;              // logical dims per lane
    constexpr int LANES_PER_PAL = 32 / N_PALETTE;
    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    // Per-WARP palette-order K/V staging (T): all WARPS columns of a tile are
    // dequantized in parallel (warp w owns column c0+w), then un-permuted to
    // logical-order k_col_tile/v_col_tile via the PalIter map. This is the
    // decode kernel's all-warp tile dequant — vs a per-column 1-2 warp stage it
    // amortizes the block sync over WARPS columns and uses every warp's loads.
    __shared__ T k_stage[WARPS_PER_BLOCK * HEAD_DIM];
    __shared__ T v_stage[WARPS_PER_BLOCK * HEAD_DIM];

    const int slot_idx = (int)blockIdx.x;
    const int kv_head_idx = (int)blockIdx.y;
    if (slot_idx >= batch || kv_head_idx >= n_kv_head) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = WARPS_PER_BLOCK;

    int num_groups = n_q_head / n_kv_head;
    if (num_groups <= 0) num_groups = 1;
    const int hpg = num_groups;
    const int head_base = kv_head_idx * hpg;

    const int q_start = (int)cu_seqlens_q[slot_idx];
    const int g_total = (int)q_lens[slot_idx];
    const int kv_len = (int)kv_lens[slot_idx];
    const int prefix_len = kv_len - g_total;
    const int kv_base = (int)cu_kvlens[slot_idx];

    // ── Forward B-head window ────────────────────────────────────────────
    // Glue rows may also attend the first `fwd_b` columns of the NEXT section
    // (B), so the glue is generated as a bridge INTO B rather than as a blind
    // continuation of A. These B columns are assumed already resident in THIS
    // slot's position_map at packed indices [kv_len, kv_len+fwd_b), with
    // col_actual_pos carrying their assembled (downstream-of-glue) positions —
    // so the existing dequant-once + per-column RoPE path handles them with no
    // special case, and the glue→B relative phase is correct by construction.
    // Clamped per-slot to what is actually present: the last section / a short B
    // yields a smaller window; b_avail==0 yields none. fwd_b==0 makes this
    // kernel BIT-IDENTICAL to the backward-only path (same column set, same mask).
    // NOTE: if B is NOT in this slot's position_map, fwd_b must be 0 — the
    // separate-slot layout needs a different staging path and is out of scope here.
    const int fwd_b = min((int)fwd_window, (int)b_avail[slot_idx]);
    const int stream_cols = kv_len + fwd_b;

    // This block handles ONE query-row tile of GLUE_G_TILE glue tokens. The tiles
    // run in parallel across gridDim.z so different tiles' prefix streams (the
    // dequant work) overlap on the SMs instead of serializing as in-block passes
    // — that is what keeps the per-column dequant ~once in wall-clock for large
    // glue, within the per-block smem budget for the flash-state.
    const int g0 = (int)blockIdx.z * GLUE_G_TILE;
    if (g0 >= g_total) return; // empty tile: this slot has fewer glue tokens

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices = slot.n_slices;
    const uint64_t slices_ptr = slot.slices_ptr;

    // ── Flash-state lives in REGISTERS, not smem ─────────────────────────
    // A row is owned by a whole warp (all 32 lanes; lane `l` owns head dims
    // [l*VEC, l*VEC+VEC)). Warp `w` owns rows {w, w+WARPS, w+2*WARPS, …}, i.e.
    // ROWS_PER_WARP = ROWS/WARPS = GLUE_G_TILE rows. Keeping Q + the online-
    // softmax accumulator (O, m, l) per-lane in registers — rather than in
    // smem like the decode/prefill never do for O — drops the per-column smem
    // read-modify-write and shrinks dynamic smem to just the column staging,
    // which is what frees occupancy (the kernel's real bottleneck).
    // ROWS = GLUE_G_TILE*8 rows max (hpg<=8), split WARPS_PER_BLOCK ways.
    constexpr int ROWS_PER_WARP = GLUE_G_TILE;     // = (GLUE_G_TILE*8) / WARPS
    constexpr int TILE_COLS = WARPS_PER_BLOCK;     // one column per warp per tile
    // Dynamic smem: only the current tile's dequanted columns (logical order).
    //   k_col : [TILE_COLS][HEAD_DIM] F32
    //   v_col : [TILE_COLS][HEAD_DIM] F32
    extern __shared__ float glue_smem[];
    float* k_col = glue_smem;                       // TILE_COLS*HEAD_DIM
    float* v_col = k_col + TILE_COLS * HEAD_DIM;    // TILE_COLS*HEAD_DIM

    // Per-lane register flash-state for this warp's rows.
    float q_reg[ROWS_PER_WARP][VEC];
    float o_reg[ROWS_PER_WARP][VEC];
    float m_reg[ROWS_PER_WARP];
    float l_reg[ROWS_PER_WARP];

    // ── New-token (glue) K/V scatter — one warp per glue token, mirroring the
    // decode kernel's fused scatter. Each glue token is written un-rotated into
    // its writer chunk (re-RoPE'd at read); glue is never a retrieval target so
    // the R16 Q region is zeroed. ──
    for (int t = warp; t < g_total; t += n_warps) {
        const int write_slice = (int)glue_write_slice[q_start + t];
        const int within = (int)glue_write_in_blk[q_start + t];
        const uint8_t* w_slice = get_slice<HEAD_DIM>(slices_ptr, write_slice, n_kv_head);
        const uint8_t* w_head = get_head<HEAD_DIM>(w_slice, kv_head_idx);
        const int pal = lane / LANES_PER_PAL;
        const int local_lane = lane % LANES_PER_PAL;
        const uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(w_head, pal);
        const uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(w_head, pal);
        if (k_ptr_p == 0) continue;
        char* k_arena = (char*)(uintptr_t)k_ptr_p;
        char* v_arena = (char*)(uintptr_t)v_ptr_p;
        const int k_fmt = kvhead_k_fmt<HEAD_DIM>(w_head, pal);
        const int v_fmt = kvhead_v_fmt<HEAD_DIM>(w_head, pal);
        const int k_esz = ArenaFormat::float_elem_size(k_fmt);
        const int v_esz = ArenaFormat::float_elem_size(v_fmt);
        const int64_t src_base =
            ((int64_t)(q_start + t) * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
        float k_regs[VEC], v_regs[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(k_new[src_base + lane * VEC + j]);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(v_new[src_base + lane * VEC + j]);
        if (k_fmt == ArenaFormat::R16) {
            float q_zero[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) q_zero[j] = 0.f;
            write_regs_to_r16<VEC>(k_arena, 0, within, local_lane, k_regs, q_zero);
        } else if (k_esz > 0) {
            write_regs_to_arena<VEC>(k_arena, (int64_t)within * SUB_HEAD_DIM, local_lane, k_esz, k_fmt, k_regs);
        }
        if (v_esz > 0) {
            write_regs_to_arena<VEC>(v_arena, (int64_t)within * SUB_HEAD_DIM, local_lane, v_esz, v_fmt, v_regs);
        }
    }
    __syncthreads();

    {
        const int g_tile = min(GLUE_G_TILE, g_total - g0);
        const int n_rows = g_tile * hpg; // (token, head) rows in this tile

        // ── Load this warp's rows' Q into registers, RoPE'd at each glue token's
        // true position; zero the register flash-state. Each lane holds VEC head
        // dims of each of its ROWS_PER_WARP rows. No sync — registers are
        // per-thread (the glue-scatter sync already ordered the writeback). ──
        #pragma unroll
        for (int rl = 0; rl < ROWS_PER_WARP; ++rl) {
            const int row = warp + rl * n_warps;
            m_reg[rl] = -1e38f;
            l_reg[rl] = 0.f;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) o_reg[rl][j] = 0.f;
            if (row >= n_rows) {
                #pragma unroll
                for (int j = 0; j < VEC; ++j) q_reg[rl][j] = 0.f;
                continue;
            }
            const int t = g0 + row / hpg;
            const int h = row % hpg;
            const int q_head = head_base + h;
            const int true_pos = (int)col_actual_pos[kv_base + prefix_len + t];
            const int64_t qb =
                ((int64_t)(q_start + t) * (int64_t)n_q_head + (int64_t)q_head) * (int64_t)HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) q_reg[rl][j] = to_f32<Q_T>(q[qb + lane * VEC + j]);
            if (rope_interleaved)
                apply_rope_interleaved_f32<VEC, HEAD_DIM>(q_reg[rl], lane, true_pos, rope_cs);
            else
                apply_rope_rotary_f32<VEC, HEAD_DIM>(q_reg[rl], lane, true_pos, rope_cs);
        }

        // ── Stream every column [0, kv_len) in TILES of WARPS columns, packed
        // order via the slot's position_map. This covers the sealed prefix AND
        // the freshly-written glue (whose columns are in the position_map but NOT
        // in the writer slices' `len`, so a per-slice scan would miss them).
        // Warp w dequants column c0+w into its slot of k_stage/v_stage — all
        // warps in parallel — then un-permutes + RoPEs it into k_col/v_col, so
        // each column is dequantized exactly once and the block syncs ONCE per
        // WARPS columns instead of twice per column. `col_actual_pos[c]` is each
        // column's true sequence position (causal mask + RoPE).
        int cur_slice = -1;             // per-warp: this warp's last-seen slice
        PalIter<VEC, HEAD_DIM> ki, vi;  // per-warp un-permute maps
        for (int c0 = 0; c0 < stream_cols; c0 += TILE_COLS) {   // was kv_len; +fwd_b covers B-head
            const int c = c0 + warp; // this warp's column
            int col_pos = 0;
            if (c < stream_cols) {                              // was kv_len
                int slice_idx = 0, in_blk = 0;
                resolve_pos(slot, c, slice_idx, in_blk);
                const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, slice_idx, n_kv_head);
                const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);
                if (slice_idx != cur_slice) {
                    ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);
                    vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);
                    cur_slice = slice_idx;
                }
                col_pos = (int)col_actual_pos[kv_base + c];

                // Stage K + V for this warp's column (palette order).
                T* k_st = k_stage + warp * HEAD_DIM;
                T* v_st = v_stage + warp * HEAD_DIM;
                for (int p = 0; p < N_PALETTE; ++p) {
                    const uint64_t kp = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
                    if (kp != 0) {
                        const int fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, p);
                        const float scale = kvhead_k_scale<HEAD_DIM>(head_ptr, p);
                        ArenaAccessor acc((const char*)(uintptr_t)kp, fmt, sub_head_stride,
                                          sub_head_stride, BLOCKS_PER_DIM, 0);
                        acc.template load_head_scaled<T, SUB_HEAD_DIM, true>(
                            k_st + p * SUB_HEAD_DIM, 0, 0, in_blk, lane, scale);
                    }
                    const uint64_t vp = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
                    if (vp != 0) {
                        const int fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
                        const float scale = kvhead_v_scale<HEAD_DIM>(head_ptr, p);
                        ArenaAccessor acc((const char*)(uintptr_t)vp, fmt, sub_head_stride,
                                          sub_head_stride, BLOCKS_PER_DIM, 0);
                        acc.template load_head_scaled<T, SUB_HEAD_DIM, true>(
                            v_st + p * SUB_HEAD_DIM, 0, 0, in_blk, lane, scale);
                    }
                }
                cp_async_commit<true>();
                cp_async_wait<0, true>();

                // Un-permute palette → logical order; RoPE K at the column's true
                // position. Each warp covers its own column across HEAD_DIM.
                float kr[VEC], vr[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) kr[j] = to_f32<T>(k_st[ki[j]]);
                #pragma unroll
                for (int j = 0; j < VEC; ++j) vr[j] = to_f32<T>(v_st[vi[j]]);
                if (rope_interleaved)
                    apply_rope_interleaved_f32<VEC, HEAD_DIM>(kr, lane, col_pos, rope_cs);
                else
                    apply_rope_rotary_f32<VEC, HEAD_DIM>(kr, lane, col_pos, rope_cs);
                #pragma unroll
                for (int j = 0; j < VEC; ++j) k_col[warp * HEAD_DIM + lane * VEC + j] = kr[j];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) v_col[warp * HEAD_DIM + lane * VEC + j] = vr[j];
            }
            __syncthreads(); // all warps' tile columns staged into k_col/v_col

            const int tile_cols = min(TILE_COLS, stream_cols - c0);  // was kv_len

            // Score the tile's columns against this warp's rows, accumulating the
            // online-softmax state in registers. `m_reg`/`l_reg` stay
            // warp-uniform (the dot is reduced across lanes), so no per-lane
            // divergence and no smem round-trip.
            #pragma unroll
            for (int rl = 0; rl < ROWS_PER_WARP; ++rl) {
                const int row = warp + rl * n_warps;
                if (row >= n_rows) continue;
                const int t = g0 + row / hpg;
                const int row_pos = (int)col_actual_pos[kv_base + prefix_len + t];
                for (int cc = 0; cc < tile_cols; ++cc) {
                    const int cabs = c0 + cc;                       // packed column index
                    const int cpos = (int)col_actual_pos[kv_base + cabs];
                    // Columns [0, kv_len)        : prefix (A) + glue — causal by
                    //                              actual position (backward look,
                    //                              unbounded over A; glue stays
                    //                              causal among itself).
                    // Columns [kv_len, stream_cols): B-head forward window — always
                    //                              visible to glue rows regardless of
                    //                              position (this is the +fwd_b look).
                    //                              Only B is opened forward; glue↔glue
                    //                              causality is unchanged.
                    const bool fwd_b_col = (cabs >= kv_len);
                    if (!fwd_b_col && cpos > row_pos) continue; // causal (actual position)
                    float dot = 0.f;
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        dot += q_reg[rl][j] * k_col[cc * HEAD_DIM + lane * VEC + j];
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, o);
                    const float score = dot * softmax_scale;

                    const float m_new = fmaxf(m_reg[rl], score);
                    const float alpha = fast_exp::exp2<float, fast_exp::Softmax>(
                        make_float2(m_reg[rl] - m_new, 0.f)).x;
                    const float beta = fast_exp::exp2<float, fast_exp::Softmax>(
                        make_float2(score - m_new, 0.f)).x;
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        o_reg[rl][j] = o_reg[rl][j] * alpha + beta * v_col[cc * HEAD_DIM + lane * VEC + j];
                    m_reg[rl] = m_new;
                    l_reg[rl] = l_reg[rl] * alpha + beta;
                }
            }
            __syncthreads(); // before the next tile overwrites k_col/v_col
        }

        // ── Normalize (register O / register l) + write out this warp's rows. ──
        #pragma unroll
        for (int rl = 0; rl < ROWS_PER_WARP; ++rl) {
            const int row = warp + rl * n_warps;
            if (row >= n_rows) continue;
            const int t = g0 + row / hpg;
            const int h = row % hpg;
            const int q_head = head_base + h;
            const float inv = __fdividef(1.f, fmaxf(l_reg[rl], 1e-10f));
            const int64_t ob =
                ((int64_t)(q_start + t) * (int64_t)n_q_head + (int64_t)q_head) * (int64_t)HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j)
                out[ob + lane * VEC + j] = from_f32<O>(o_reg[rl][j] * inv);
        }
    }
}

// Host launcher. One block per (slot, kv_head, glue_tile); the flash-state is
// register-resident, so dynamic smem is just the tile's dequanted K/V columns.
// HEAD_DIM=128 is the production path.
template <typename Q_T, typename T, typename O, int HEAD_DIM>
inline void launch_paged_glue_attn(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int batch,
    int max_glue, // max q_lens[b] over slots — sizes the parallel glue-tile grid
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const float* rope_cs,
    int rope_interleaved,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    const uint32_t* col_actual_pos,
    const uint32_t* cu_kvlens,
    const uint32_t* glue_write_slice,
    const uint32_t* glue_write_in_blk,
    uint32_t fwd_window,            // forward B-head window (tokens); 0 == backward-only
    const uint32_t* b_avail,        // per-slot count of B columns resident after kv_len
    cudaStream_t stream
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int TILE_COLS = WARPS_PER_BLOCK;
    // Flash-state is register-resident now; dynamic smem is only the tile's
    // dequanted columns (k_col + v_col). Independent of GLUE_G_TILE.
    const size_t smem_bytes = (size_t)(2 * TILE_COLS * HEAD_DIM) * sizeof(float);

    auto kern = paged_glue_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK>;
    int dev = 0;
    cudaGetDevice(&dev);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaError_t e1 =
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

    int glue_tiles = (max_glue + GLUE_G_TILE - 1) / GLUE_G_TILE;
    if (glue_tiles < 1) glue_tiles = 1;
    dim3 grid(batch, n_kv_head, glue_tiles);
    dim3 block(WARPS_PER_BLOCK * 32);
    kern<<<grid, block, smem_bytes, stream>>>(
        q, headers_ptr, out, batch, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, rope_interleaved != 0,
        cu_seqlens_q, q_lens, kv_lens, col_actual_pos, cu_kvlens,
        glue_write_slice, glue_write_in_blk,
        fwd_window, b_avail);
    cudaError_t e2 = cudaGetLastError();
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        printf("[GLUE LAUNCH] smem_req=%zu dev_max=%d setattr=%s launch=%s\n",
               smem_bytes, max_smem, cudaGetErrorString(e1), cudaGetErrorString(e2));
    }
}

} // namespace paged_glue
