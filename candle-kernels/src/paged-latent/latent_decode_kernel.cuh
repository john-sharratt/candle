#pragma once
// Paged latent-attention DECODE: single-query-per-slot hybrid window+compressed
// attention with fused FP8 scatter. Shared helpers/pool/combine live in
// latent_common.cuh; the batched prefill entry is in latent_prefill_kernel.cuh.
#include "latent_common.cuh"

namespace latent_attn {

// =============================================================================
// Decode kernel.
//
// grid  = (num_slots, head_tiles = ceil(H / 16), num_splits)
// block = 256 (8 warps)
//
// (Measured dead ends — do not revisit without new hardware: 32-head/512-
// thread blocks, with or without double-buffered key staging, are FLAT to
// slightly worse than this shape once the int8 corpus cache landed — the
// key walk is L2-hot and the grid's 640 independent blocks already hide the
// load latency; halving the walk or overlapping it buys nothing.)
//
// Thread roles:
//   Q stage    : thread t owns (head_local = t/16, 32 dims at (t%16)*32).
//   Key stage  : warp w owns key w of the tile; lane owns 16 dims (lane*16).
//   QK         : the WARPS(8) warps stride the NPAL(16) bands (p = warp,
//                warp+8), SUB/32 = 1 m16n8k32 k-step each, int32-accumulated
//                (one scale per band), scaled into scores_p[p][16][8]; the
//                softmax owner sums the NPAL bands.
//   softmax/PV : warp w owns heads {2w, 2w+1}; lane owns 16 dims per head
//                (out accumulator = 32 f32/thread).
// =============================================================================
// Identity band layout only (dim d → band d/SUB). No palette-map routing.
template <typename T, int HEAD_DIM, int ROPE_DIM>
__global__ void __launch_bounds__(WARPS * 32, 4)
latent_decode_kernel(
    const T* __restrict__ q,           // [slots, H, HEAD_DIM] pre-RoPE
    const uint8_t* __restrict__ headers,
    const T* __restrict__ kv_new,      // [slots, HEAD_DIM] pre-RoPE latent
    const int8_t* __restrict__ nope_i8,   // [G_total, NOPE_DIM] nope int8
    const float* __restrict__ nope_scale, // [G_total, NOPE_BANDS] per-nope-band scale
    const T* __restrict__ comp_rope,      // [G_total, ROPE_DIM] rope PRE-rotation bf16
    const uint32_t* __restrict__ comp_idx, // [slots, max_sel] ascending GIDs
    const uint32_t* __restrict__ comp_cnt, // [slots]
    const uint32_t* __restrict__ comp_pos, // [G_total] assembled position per entry
                                           // (rope-at-load: rotate the rope bands here)
    const uint32_t* __restrict__ q_pos_in, // [slots] query position (explicit, so the
                                           // windowless slot needs no writer slice)
    const float* __restrict__ rope_tab,    // factored cos/sin table (common)
    float* __restrict__ partial_acc,   // [slots*H, splits, HEAD_DIM]
    float* __restrict__ partial_ml,    // [slots*H, splits, 2]
    int num_slots,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    // Nullable stage-dump (slot 0 / head-tile 0 / split 0 / tile 0 only), for
    // the mirror oracle's stage-by-stage comparison. Section offsets are
    // NPAL-parameterized (see DBG_* below); at HEAD_DIM=512, KEYS_TILE=8:
    //   scaleQ[16][NPAL] | sQ[16][512] | scaleK[8][NPAL] | sK[8][512]
    //   | kv_f[8][512] (roped, staged) | summed logits[16][8]
    float* __restrict__ dbg
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    static_assert(HEAD_DIM % NPAL == 0 && SUB % 32 == 0, "bands must be 32-dim MMA chunks");
    static_assert(ROPE_DIM % 2 == 0, "interleaved RoPE needs even rope dim");

    // Stage-dump section offsets (NPAL-parameterized so the mirror oracle's
    // read offsets track the band count). The dump is always 16-head-sized.
    constexpr int DBG_SCALEQ = 0;                              // [16][NPAL]
    constexpr int DBG_SQ = DBG_SCALEQ + 16 * NPAL;             // [16][HEAD_DIM]
    constexpr int DBG_SCALEK = DBG_SQ + 16 * HEAD_DIM;         // [KEYS_TILE][NPAL]
    constexpr int DBG_SK = DBG_SCALEK + KEYS_TILE * NPAL;      // [KEYS_TILE][HEAD_DIM]
    constexpr int DBG_KVF = DBG_SK + KEYS_TILE * HEAD_DIM;     // [KEYS_TILE][HEAD_DIM]
    constexpr int DBG_LOGITS = DBG_KVF + KEYS_TILE * HEAD_DIM; // [16][KEYS_TILE]

    const int slot_idx = (int)blockIdx.x;
    const int head_base = (int)blockIdx.y * HEADS_TILE;
    const int split_idx = (int)blockIdx.z;
    const int num_splits = (int)gridDim.z;
    const int tid = (int)threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (slot_idx >= num_slots) return;

    // Per-thread flash state: heads {2*warp, 2*warp+1} × dims [lane*16, +16).
    constexpr int DPT = HEAD_DIM / 32;  // dims per thread per head (16)
    // A comp lane owns DPT consecutive dims; the two-region read selects int8
    // (nope) vs bf16 (rope) per lane on the assumption that no lane straddles the
    // [0,NOPE_DIM)/[NOPE_DIM,HEAD_DIM) boundary — i.e. DPT divides NOPE_DIM.
    static_assert(NOPE_DIM % DPT == 0, "DPT must divide NOPE_DIM (no lane straddles the nope/rope split)");
    float m_i[2] = {-1e38f, -1e38f};
    float l_i[2] = {0.f, 0.f};
    float out_reg[2][DPT];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int j = 0; j < DPT; ++j) out_reg[h][j] = 0.f;

    // Emit this block's split partial for its two heads (always runs — the
    // combine kernel is unconditional).
    auto emit_partials = [&]() {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int head = head_base + 2 * warp + h;
            if (head >= n_q_head) continue;
            int64_t base = ((int64_t)slot_idx * n_q_head + head) * num_splits + split_idx;
            float* acc = partial_acc + base * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) acc[lane * DPT + j] = out_reg[h][j];
            if (lane == 0) {
                partial_ml[base * 2] = m_i[h];
                partial_ml[base * 2 + 1] = l_i[h];
            }
        }
    };

    const SlotHeader& slot = get_slot_header(headers, slot_idx);
    const uint32_t n_slices = slot.n_slices;
    const uint64_t slices_ptr = slot.slices_ptr;
    const uint32_t n_sel = comp_cnt ? comp_cnt[slot_idx] : 0;

    // Query position is explicit — the windowless pure-substrate slot has no
    // writer slice to derive it from. INVARIANT (caller-enforced): when a window
    // ring exists this MUST equal the writer slice's implied position,
    // `slice_rope(write_slice) + ws_len` — the window keys rope at slice-derived
    // positions, so a mismatched q_pos would attend them at the wrong relative
    // distance and mis-place the just-scattered token in the causal test. The
    // wave passes the same `decode_pos[i]` the writer slice was set up for.
    const int q_pos = (int)q_pos_in[slot_idx];

    // Attend when EITHER source has keys: a windowless slot (n_slices==0, no
    // window ring) still attends its selected compressed set. NOTE: live decode
    // always pre-allocates a writer chunk (n_slices>=1, see
    // batched_inference build_decode_metadata), so this is defensive — a true
    // windowless slot would also need the current token's `kv_new` stored
    // elsewhere, since the fused scatter below (gated on n_slices>0) skips it.
    if (n_slices == 0 && n_sel == 0) {
        emit_partials();
        return;
    }

    // ─── Fused single-latent scatter (warp 0): write this token's pre-RoPE
    // latent into the writer chunk's FP8 band arenas. K≡V → K bands only. Only
    // when a window ring exists (n_slices>0) — the windowless slot writes no
    // local chunk. n_slices is block-uniform, so the __syncthreads after is
    // reached by every thread. ───
    if (n_slices > 0) {
        uint8_t* write_slice_ptr =
            get_slice_mut<HEAD_DIM>(slices_ptr, (int)slot.write_slice, 1);
        const int ws_offset = (int)slice_offset(write_slice_ptr);
        const int ws_len = (int)slice_len(write_slice_ptr);
        const int within = ws_offset + ws_len;
        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, 0);
            const T* src = kv_new + (int64_t)slot_idx * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) {
                int d = lane * DPT + j;
                int band = d / SUB;
                int in_band = d % SUB;
                uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM, NPAL>(head_ptr, band);
                if (band_ptr != 0) {
                    const int fmt = kvhead_k_fmt<HEAD_DIM, NPAL>(head_ptr, band);
                    const float outer =
                        kvhead_k_scale<HEAD_DIM, NPAL>(head_ptr, band);
                    store_band_elem<SUB>(band_ptr, fmt, outer, within, in_band,
                                         to_f32<T>(src[d]));
                }
            }
        }
    }
    __syncthreads();

    // ─── Shared memory ────────────────────────────────────────────────────
    __shared__ alignas(128) int8_t sQ[HEADS_TILE][HEAD_DIM];       // 8 KB
    __shared__ alignas(16) float scaleQ[HEADS_TILE][NPAL];
    __shared__ alignas(128) T kv_f[KEYS_TILE][HEAD_DIM];           // 8 KB (bf16)
    __shared__ alignas(128) int8_t sK[KEYS_TILE][HEAD_DIM];        // 4 KB
    __shared__ alignas(16) float scaleK[KEYS_TILE][NPAL];
    __shared__ alignas(16) float scores_p[NPAL][HEADS_TILE][KEYS_TILE]; // 2 KB
    __shared__ int key_valid[KEYS_TILE];

    // ─── Q stage: load 16 heads, RoPE at q_pos, per-band int8 quant ───────
    {
        const int head_local = tid / 16;        // 0..15
        const int dseg = (tid % 16) * 32;       // 32 dims per thread
        const int head = head_base + head_local;
        float qr[32];
        if (head < n_q_head) {
            const T* qp = q + ((int64_t)slot_idx * n_q_head + head) * HEAD_DIM + dseg;
            #pragma unroll
            for (int j = 0; j < 32; ++j) qr[j] = to_f32<T>(qp[j]);
        } else {
            #pragma unroll
            for (int j = 0; j < 32; ++j) qr[j] = 0.f;
        }
        // Interleaved RoPE on dims >= NOPE_DIM (pairs are thread-local: the
        // 32-dim segment is even-aligned).
        #pragma unroll
        for (int j = 0; j < 32; j += 2) {
            int d = dseg + j;
            if (d >= NOPE_DIM) {
                rope_pair<ROPE_DIM / 2>(qr[j], qr[j + 1], rope_tab, q_pos,
                                        (d - NOPE_DIM) >> 1);
            }
        }
        // Band max over the SUB/32 threads covering (head, band). Each Q
        // thread owns 32 dims; a band spans SUB dims = SUB/32 adjacent threads,
        // so the reduction and the write predicate scale with the band width.
        constexpr int Q_THREADS_PER_BAND = SUB / 32;
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(qr[j]));
        #pragma unroll
        for (int off = 1; off < Q_THREADS_PER_BAND; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        // Explicit reciprocal multiply (NOT `/ 127.f`): nvcc lowers constant
        // division to this multiply regardless of -prec-div, so the mirror
        // contract writes the op both sides compute.
        float s = __fdiv_rn(mx, 127.f); // IEEE division (mirror parity)
        if (s == 0.f) s = 1.f;
        const int band = (tid % 16) / Q_THREADS_PER_BAND;
        if ((tid & (Q_THREADS_PER_BAND - 1)) == 0) scaleQ[head_local][band] = s;
        float inv = __frcp_rn(s); // IEEE reciprocal (mirror parity under fast-math)
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float v = fminf(fmaxf(qr[j] * inv, -127.f), 127.f);
            sQ[head_local][dseg + j] = (int8_t)__float2int_rn(v);
        }
    }
    __syncthreads();

    // ─── Window tiling (gap-aware slice walk, forked from paged-decode) ───
    auto slice_eff_len = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, 1);
        int len = (int)slice_len(sl);
        int off = (int)slice_offset(sl);
        if (s == (int)slot.write_slice && len < CHUNK_SIZE && off + len < CHUNK_SIZE)
            len += 1;  // the just-scattered token
        return len;
    };
    auto slice_tiles = [&](int s) -> int {
        return (slice_eff_len(s) + KEYS_TILE - 1) / KEYS_TILE;
    };
    auto tile_to_slice = [&](int tile_idx, int& slice_out, int& within_base_out) {
        int base = 0, s = 0;
        while (s + 1 < (int)n_slices) {
            int st = slice_tiles(s);
            if (base + st <= tile_idx) { base += st; ++s; } else break;
        }
        slice_out = s;
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, 1);
        within_base_out = (int)slice_offset(sl) + (tile_idx - base) * KEYS_TILE;
    };

    int n_win_tiles = 0;
    for (int s = 0; s < (int)n_slices; ++s) n_win_tiles += slice_tiles(s);
    const int n_comp_tiles = ((int)n_sel + KEYS_TILE - 1) / KEYS_TILE;
    const int n_tiles = n_win_tiles + n_comp_tiles;

    const int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
    int tile_lo = split_idx * tiles_per_split;
    int tile_hi = tile_lo + tiles_per_split;
    if (tile_lo > n_tiles) tile_lo = n_tiles;
    if (tile_hi > n_tiles) tile_hi = n_tiles;

    // ─── Load one tile's 8 keys into kv_f/sK (warp w = key w) ────────────
    // Stage key `warp` of tile `tile_idx` (one warp per key).
    auto load_tile = [&](int tile_idx) {
        const int key = warp;
        bool valid = false;
        int key_pos = 0;
        float regs[DPT];
        if (tile_idx < n_win_tiles) {
            // Window source: band arenas, format-dispatched per band (FP8
            // writer chunk; adaptive quant on policy-compressed sealed chunks).
            int sl_idx, within_base;
            tile_to_slice(tile_idx, sl_idx, within_base);
            int within = within_base + key;
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, sl_idx, 1);
            int off = (int)slice_offset(sl);
            if (sl_idx < (int)n_slices && within < off + slice_eff_len(sl_idx)) {
                key_pos = (int)slice_rope(sl) + (within - off);
                // Sliding-window + causal bound (exact regardless of chunk
                // granularity).
                if (key_pos <= q_pos && key_pos > q_pos - window_size) {
                    valid = true;
                    const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, 0);
                    // Identity layout: the lane's DPT consecutive dims sit in
                    // ONE band (DPT | SUB), so {ptr, fmt, outer} resolve once
                    // and the elements dispatch on the format tag.
                    auto ident_read = [&] {
                        const int band = (lane * DPT) / SUB;
                        const uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM, NPAL>(head_ptr, band);
                        const int fmt = kvhead_k_fmt<HEAD_DIM, NPAL>(head_ptr, band);
                        const float outer = kvhead_k_scale<HEAD_DIM, NPAL>(head_ptr, band);
                        if (band_ptr && fmt == ArenaFormat::F8E4M3) {
                            // Hot path: the direct FP8 row read, format check
                            // hoisted.
                            const uint8_t* src = (const uint8_t*)(uintptr_t)band_ptr;
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) {
                                int d = lane * DPT + j;
                                regs[j] =
                                    fp8_to_f32(src[(int64_t)within * SUB + (d % SUB)]) / outer;
                            }
                        } else if (band_ptr) {
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) {
                                int d = lane * DPT + j;
                                regs[j] = load_band_elem<SUB>(band_ptr, fmt, outer, within, d % SUB);
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
                        }
                    };
                    // Identity band layout is the only layout: dim d lives in
                    // band d/SUB (no palette regroup / pal_map).
                    ident_read();
                }
            }
        } else {
            // Compressed source: the persistent POSITION-FREE two-region corpus
            // cache (nope int8 + per-band scale; rope bf16 pre-rotation). Read
            // both regions, set the key's position to this entry's ASSEMBLED
            // position, and let the shared rope-at-load below rotate the rope
            // bands — the same path the window keys take, so the cache carries no
            // baked position and survives re-selection.
            constexpr int NOPE_BANDS = NOPE_DIM / SUB;
            int e = (tile_idx - n_win_tiles) * KEYS_TILE + key;
            if (e < (int)n_sel) {
                uint32_t gid = comp_idx[(int64_t)slot_idx * max_sel + e];
                if (gid != 0xFFFFFFFFu) {
                    key_pos = (int)comp_pos[gid];
                    // Causal guard: a compressed entry must not sit in the
                    // query's future (a reassembly/selection bug could place it
                    // there). Drop it rather than attend a future key.
                    if (key_pos <= q_pos) {
                        valid = true;
                        const int8_t* nsrc = nope_i8 + (int64_t)gid * NOPE_DIM;
                        const float* scl = nope_scale + (int64_t)gid * NOPE_BANDS;
                        const T* rsrc = comp_rope + (int64_t)gid * ROPE_DIM;
                        // A comp lane owns DPT consecutive dims, all on one side of
                        // the nope/rope boundary (DPT | NOPE_DIM), so no per-dim
                        // divergence: nope → int8·scale, rope → bf16.
                        #pragma unroll
                        for (int j = 0; j < DPT; ++j) {
                            int d = lane * DPT + j;
                            regs[j] = (d < NOPE_DIM)
                                ? (float)nsrc[d] * scl[d / SUB]
                                : to_f32<T>(rsrc[d - NOPE_DIM]);
                        }
                    }
                }
            }
        }
        if (!valid) {
            #pragma unroll
            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
        }
        // RoPE at the key's own position (pairs are lane-local: 16-dim segments
        // are even-aligned). Both window keys and compressed entries rotate here
        // — the compressed rope bands are stored pre-rotation and rotated at this
        // entry's assembled position (`key_pos = comp_pos[gid]`).
        if (valid) {
            #pragma unroll
            for (int j = 0; j < DPT; j += 2) {
                int d = lane * DPT + j;
                if (d >= NOPE_DIM) {
                    rope_pair<ROPE_DIM / 2>(regs[j], regs[j + 1], rope_tab,
                                            key_pos, (d - NOPE_DIM) >> 1);
                }
            }
        }
        // Stage FP latent (the PV read; K≡V) + per-band int8 (the QK read).
        #pragma unroll
        for (int j = 0; j < DPT; ++j)
            kv_f[key][lane * DPT + j] = from_f32<T>(regs[j]);
        {
            float mx = 0.f;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) mx = fmaxf(mx, fabsf(regs[j]));
            // A K lane owns DPT dims; a band spans SUB dims = SUB/DPT adjacent
            // lanes (lanes [KLB*b, KLB*b+KLB) cover band b), so the reduction
            // and write predicate scale with the band width.
            constexpr int KLB = SUB / DPT;  // lanes per band
            #pragma unroll
            for (int off = 1; off < KLB; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            // Explicit reciprocal multiply (NOT `/ 127.f`): nvcc lowers constant
        // division to this multiply regardless of -prec-div, so the mirror
        // contract writes the op both sides compute.
        float s = __fdiv_rn(mx, 127.f); // IEEE division (mirror parity)
            if (s == 0.f) s = 1.f;
            if ((lane & (KLB - 1)) == 0) scaleK[key][lane / KLB] = s;
            float inv = __frcp_rn(s); // IEEE reciprocal (mirror parity under fast-math)
            #pragma unroll
            for (int j = 0; j < DPT; ++j) {
                float v = fminf(fmaxf(regs[j] * inv, -127.f), 127.f);
                sK[key][lane * DPT + j] = (int8_t)__float2int_rn(v);
            }
        }
        if (lane == 0) key_valid[key] = valid ? 1 : 0;
    };

    const bool dump = dbg != nullptr && slot_idx == 0 && blockIdx.y == 0 && split_idx == 0;

    // ─── Main tile loop ───────────────────────────────────────────────────
    for (int tile = tile_lo; tile < tile_hi; ++tile) {
        load_tile(tile);
        __syncthreads();

        if (dump && tile == tile_lo && tid == 0) {
            // The dump layout is fixed at 16 heads (the mirror oracle's
            // documented offsets); dump the block's first m16 row-tile.
            for (int h = 0; h < 16; ++h)
                for (int p = 0; p < NPAL; ++p) dbg[DBG_SCALEQ + h * NPAL + p] = scaleQ[h][p];
            for (int h = 0; h < 16; ++h)
                for (int d = 0; d < HEAD_DIM; ++d)
                    dbg[DBG_SQ + h * HEAD_DIM + d] = (float)sQ[h][d];
            for (int t = 0; t < KEYS_TILE; ++t)
                for (int p = 0; p < NPAL; ++p) dbg[DBG_SCALEK + t * NPAL + p] = scaleK[t][p];
            for (int t = 0; t < KEYS_TILE; ++t)
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dbg[DBG_SK + t * HEAD_DIM + d] = (float)sK[t][d];
                    dbg[DBG_KVF + t * HEAD_DIM + d] = to_f32<T>(kv_f[t][d]);
                }
        }

        // QK: the WARPS warps cover the NPAL bands, one band per warp per
        // iteration; when NPAL > WARPS each warp strides `WARPS` bands (2 at
        // NPAL=16). Each band: SUB/32 k-steps of m16n8k32 int32-accumulated
        // (uniform scale within the band), then scaled into per-band float
        // partial scores. The softmax owner sums the NPAL bands.
        for (int p = warp; p < NPAL; p += WARPS) {
            int32_t c[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int ks = 0; ks < SUB / 32; ++ks) {
                uint32_t a_frag[4];
                uint32_t b_frag[2];
                fused_attn::load_a_frag_m16k32(a_frag, &sQ[0][p * SUB + ks * 32], HEAD_DIM, lane);
                fused_attn::load_b_frag_n8k32(b_frag, &sK[0][p * SUB + ks * 32], HEAD_DIM, lane);
                fused_attn::mma_int8_m16n8k32(c, a_frag, b_frag, c);
            }
            // C layout: lane holds rows {lane>>2, (lane>>2)+8}, cols
            // {(lane&3)*2, (lane&3)*2+1}.
            const int r0 = lane >> 2;
            const int c0 = (lane & 3) * 2;
            scores_p[p][r0][c0]     = (float)c[0] * scaleQ[r0][p] * scaleK[c0][p];
            scores_p[p][r0][c0 + 1] = (float)c[1] * scaleQ[r0][p] * scaleK[c0 + 1][p];
            scores_p[p][r0 + 8][c0]     = (float)c[2] * scaleQ[r0 + 8][p] * scaleK[c0][p];
            scores_p[p][r0 + 8][c0 + 1] = (float)c[3] * scaleQ[r0 + 8][p] * scaleK[c0 + 1][p];
        }
        __syncthreads();

        if (dump && tile == tile_lo && tid == 0) {
            // First m16 row-tile only — the dump layout is 16-head-sized.
            for (int h = 0; h < 16; ++h)
                for (int t = 0; t < KEYS_TILE; ++t) {
                    float s = 0.f;
                    #pragma unroll
                    for (int p = 0; p < NPAL; ++p) s += scores_p[p][h][t];
                    dbg[DBG_LOGITS + h * KEYS_TILE + t] = s;
                }
        }

        // Softmax + PV for this warp's two heads.
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int head_local = 2 * warp + h;
            if (head_base + head_local >= n_q_head) continue;
            float sc[KEYS_TILE];
            float tile_max = -1e38f;
            #pragma unroll
            for (int t = 0; t < KEYS_TILE; ++t) {
                float lg = 0.f;
                #pragma unroll
                for (int p = 0; p < NPAL; ++p) lg += scores_p[p][head_local][t];
                sc[t] = key_valid[t] ? lg * softmax_scale : -1e38f;
                tile_max = fmaxf(tile_max, sc[t]);
            }
            float new_m = fmaxf(m_i[h], tile_max);
            float alpha = ds_exp(m_i[h] - new_m);
            l_i[h] *= alpha;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) out_reg[h][j] *= alpha;
            #pragma unroll
            for (int t = 0; t < KEYS_TILE; ++t) {
                float beta = (sc[t] > -1e37f) ? ds_exp(sc[t] - new_m) : 0.f;
                l_i[h] += beta;
                #pragma unroll
                for (int j = 0; j < DPT; ++j) {
                    float v = to_f32<T>(kv_f[t][lane * DPT + j]);
                    out_reg[h][j] = __fmaf_rn(beta, v, out_reg[h][j]);
                }
            }
            m_i[h] = new_m;
        }
        __syncthreads();
    }

    emit_partials();
}

// =============================================================================
// Combine: merge split partials, fold the per-head sink, normalize, de-rotate
// the output's rope dims at the query position, write the final output.
// One block per (slot, head) row; HEAD_DIM threads.
// =============================================================================
template <typename O, int HEAD_DIM, int ROPE_DIM>
__global__ void latent_combine_kernel(
    O* __restrict__ out,                    // [slots, H, HEAD_DIM]
    const float* __restrict__ partial_acc,  // [rows, splits, HEAD_DIM]
    const float* __restrict__ partial_ml,   // [rows, splits, 2]
    const uint32_t* __restrict__ q_pos_in,  // [slots] query position (explicit)
    const float* __restrict__ sinks,        // [H]
    const float* __restrict__ rope_tab,     // factored cos/sin table (common)
    int num_rows,
    int n_q_head,
    int num_splits
) {
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    const int row = (int)blockIdx.x;
    if (row >= num_rows) return;
    const int d = (int)threadIdx.x;
    if (d >= HEAD_DIM) return;
    const int slot_idx = row / n_q_head;
    const int head = row % n_q_head;

    const float* ml = partial_ml + (int64_t)row * num_splits * 2;
    const float* pa = partial_acc + (int64_t)row * num_splits * HEAD_DIM;

    float gm = -1e38f;
    for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, ml[s * 2]);

    // Sink fold: the sink is an extra softmax column (zero value) that
    // participates in the max. Natural-e domain throughout.
    const float sink = sinks[head];
    const float m_fin = fmaxf(gm, sink);

    float acc = 0.f, L = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = ml[s * 2];
        if (!(m_s > -1e37f)) continue;  // null partial
        float w = ds_exp(m_s - m_fin);
        acc = __fmaf_rn(pa[(int64_t)s * HEAD_DIM + d], w, acc);
        L = __fmaf_rn(ml[s * 2 + 1], w, L);
    }
    L += ds_exp(sink - m_fin);

    // Exact IEEE division (not __fdividef): the combine runs once per row, and
    // the CPU mirror oracle must reproduce this bit-for-bit.
    float val = acc / fmaxf(L, 1e-10f);

    // De-rotation (inverse/conjugate rotation) at the query position. Pairs
    // (2k, 2k+1) are lane-adjacent, so the partner rides one shfl.
    if (d >= NOPE_DIM) {
        const int q_pos = (int)q_pos_in[slot_idx];
        float partner = __shfl_xor_sync(0xffffffff, val, 1);
        float c, s;
        rope_lookup<ROPE_DIM / 2>(rope_tab, q_pos, (d - NOPE_DIM) >> 1, s, c);
        // inverse: even' = x0·c + x1·s ; odd' = x1·c − x0·s (explicit-rounded)
        val = (d & 1) == 0
            ? __fadd_rn(__fmul_rn(val, c), __fmul_rn(partner, s))
            : __fsub_rn(__fmul_rn(val, c), __fmul_rn(partner, s));
    }

    out[(int64_t)row * HEAD_DIM + d] = from_f32<O>(val);
}

template <typename T, int HEAD_DIM, int ROPE_DIM>
void launch_latent_decode(
    const T* q,
    const uint8_t* headers,
    T* out,
    const T* kv_new,
    const int8_t* nope_i8,    // two-region cache: nope int8 [G, NOPE_DIM]
    const float* nope_scale,  // per-nope-band scales [G, NOPE_BANDS]
    const T* comp_rope,       // rope pre-rotation bf16 [G, ROPE_DIM]
    const uint32_t* comp_idx,
    const uint32_t* comp_cnt,
    const uint32_t* comp_pos, // [G] assembled position per entry (rope-at-load)
    const uint32_t* q_pos,    // [num_slots] query position (explicit)
    const float* sinks,
    const float* rope_tab,
    // Caller-owned split-KV partial workspace with capacity for the resolved
    // split factor: acc `[slots*H, num_splits, HEAD_DIM]`, ml `[slots*H,
    // num_splits, 2]`. Caller ownership is what makes launches lock-free.
    float* pa,
    float* pm,
    int num_slots,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int num_splits,  // resolved by the caller (latent_decode_num_splits)
    bool commit_write_len,  // advance the header write-len on-device (live buffer)
    cudaStream_t stream,
    float* dbg = nullptr    // nullable stage-dump (see kernel doc)
) {
    if (num_slots <= 0 || n_q_head <= 0 || num_splits < 1) return;
    const int head_tiles = (n_q_head + HEADS_TILE - 1) / HEADS_TILE;

    // Give shared memory the largest carveout so it stops capping occupancy
    // before the warp/register limits do (the kernel is smem-load-latency
    // bound — more resident blocks hide the stalls). Set once per process.
    static bool carveout_set = false;
    if (!carveout_set) {
        cudaFuncSetAttribute(
            (const void*)latent_decode_kernel<T, HEAD_DIM, ROPE_DIM>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        carveout_set = true;
    }

    dim3 grid(num_slots, head_tiles, num_splits);
    dim3 block(WARPS * 32);
    latent_decode_kernel<T, HEAD_DIM, ROPE_DIM><<<grid, block, 0, stream>>>(
        q, headers, kv_new, nope_i8, nope_scale, comp_rope, comp_idx, comp_cnt,
        comp_pos, q_pos, rope_tab, pa, pm, num_slots, n_q_head, softmax_scale,
        window_size, max_sel, dbg);

    const int num_rows = num_slots * n_q_head;
    latent_combine_kernel<T, HEAD_DIM, ROPE_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
        out, pa, pm, q_pos, sinks, rope_tab, num_rows, n_q_head, num_splits);

    // On-device write-len advance: only the live-buffer decode path relies on
    // this (each step reads the length the previous step committed). The wave
    // hands the kernel a private per-token header snapshot with the length
    // already patched host-side, so the commit would only touch a throwaway
    // copy — skip the launch entirely there.
    if (commit_write_len) {
        constexpr int COMMIT_THREADS = 128;
        dim3 commit_grid((num_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
        commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
            headers, num_slots, 1);
    }
}
}  // namespace latent_attn
