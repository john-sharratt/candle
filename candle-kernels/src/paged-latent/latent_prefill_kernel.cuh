#pragma once
// Paged latent-attention PREFILL — TENSOR-CORE PV.
//
// Both matmuls run on int8 tensor cores: QK (Q·Kᵀ) and PV (P·Vᵀ). All 64 MQA
// heads of one query live in a single 512-thread block so each key is loaded /
// RoPE'd / quantized once per head-pass (flash). The 64 heads are covered in
// two sequential HEAD-PASSES of 32 heads each: a 512-thread block cannot hold
// all 64 heads' 512-dim output in registers (32768 f32 / 512 threads = 64
// regs/thread → spill), so each pass computes 32 heads, EMITS its output to the
// combine buffers, then reuses the same 32-register PV accumulator for the next
// pass. The tile loop runs once per pass; sK/sVt are rebuilt per pass from the
// L2-resident int8 keys (cheap). See docs/deepseek_prefill_tensorcore_pv.md.
//
//   Q stage   : the pass's 32 heads, RoPE at my_pos, per-band scaleQ. There is
//               NO sQ — the QK A-fragment is built per pass straight from L2
//               (Q is tile-invariant), freeing the smem for 2 blocks/SM.
//   load      : warp w stages keys {w, w+16} of the 32-key tile (lane owns 16
//               dims); sK[key][dim] (per-band-per-key int8, for QK).
//   Vt build  : sVt[dim][key] holds the PV operand with a scale constant over
//               keys (so it factors out of the contraction). Comp tiles GATHER
//               it pre-quantized from comp_v8 (per-dim-global scale, built by
//               the corpus pre-pass — no per-tile max/requant, epilogue scale
//               is the kernel-constant comp_vmax); window/fresh tiles requant
//               from sK with a per-tile per-dim scale.
//   QK        : warps 0-15 own band p = warp (one band each), covering BOTH
//               row-tiles of the pass's 32 heads; SUB/32 = 1 m16n8k32 per band,
//               atomic-summed into scores[head][key].
//   softmax   : warp w owns pass-heads {2w,2w+1}; lane l owns key l; online m/l
//               and P=exp(sc-m) quantized ×127 into s_p8[head][key]; alpha→smem.
//   PV        : warp = (row_tile w>>3, dim_group w&7) → 16 heads × 64 dims;
//               per 8-dim n-slice, mma(s_p8[row_tile], sVt[dim_group]) → o_acc,
//               rescaled by alpha each tile.
//   emit      : per pass, softmax warp writes (m,l); PV warp writes o_acc (ΣpV).
//               The combine kernel folds the sink, normalizes, and de-rotates.
//
// Numerics: int8 P·V is a NEW PV (vs the bf16 FMA), so this trades the
// prefill↔decode bit-exactness for the tolerance gate (harness float-ref +
// prefill_chunked + wave/StoryRewrite argmax). Int8-P adds ~0.4% on the
// int8-QK envelope.
//
// SECOND prefill↔decode divergence, on the COMPRESSED path: the two corpus
// builders no longer produce the same int8 for an entry. Prefill BAKES RoPE
// (rotate, then per-band component amax on the rotated values —
// `latent_rope_quant_corpus_kernel`); decode is POSITION-FREE (pair-magnitude
// bound on the pre-rotation values, then dequant → rotate → requant with a fresh
// amax — `latent_quant_corpus_range_kernel`). Same entry, same position → a
// slightly different int8 and scale. This stays inside the tolerance gate above,
// but it is a distinct source from the int8-PV delta — a compressed-path
// tolerance regression should be looked for HERE, not in the PV.
#include "latent_common.cuh"

namespace latent_attn {

constexpr int PF_HEADS = 64;   // query heads per block (n_q_head must be ≤ this)
constexpr int PF_WARPS = 16;   // 512-thread block
constexpr int PF_KEYS = 32;    // keys per tile = softmax lane width
// PV operands (s_p8, sVt) carry a +16 key pad so ldmatrix loads the 8 tile
// rows into distinct banks (a bare 32-stride is a multiple of 16 but conflicts
// 2-way in ldmatrix — paged-prefill's V8T_LD = TILE_TOK+16 trick).
constexpr int PF_KPAD = PF_KEYS;                 // 32 — s_p8 pad (A-operand, loaded
                                                 // once/tile): bare 16B-aligned
                                                 // stride, minor 2-way conflict.
constexpr int PF_VPAD = PF_KEYS + 16;            // 48 — sVt pad (PV B-operand,
                                                 // the HOT ldmatrix): +16 keeps
                                                 // the 8 rows in distinct banks
                                                 // (no conflict) while the block
                                                 // still fits 2/SM.
// scores row stride in WORDS: PF_KEYS would be ≡ 0 (mod 32 banks), aliasing
// every row onto bank 0 so the QK atomicAdd band-collapse serializes on 8
// banks. +1 word staggers rows across all 32 banks (row r starts at bank r).
constexpr int PF_SCR_LD = PF_KEYS + 1;           // 33
constexpr int PF_ROW_TILES = 2;                  // m16 row-tiles per head-pass
constexpr int PF_PASS_HEADS = PF_ROW_TILES * 16; // 32 heads processed per pass
constexpr int PF_PASSES = PF_HEADS / PF_PASS_HEADS; // 2 head-passes
constexpr int PF_DGROUPS = 8;                    // output-dim groups (512/64)
constexpr int PF_GDIMS = 512 / PF_DGROUPS;       // 64 dims per group (HEAD_DIM=512)

// Dynamic-smem byte size for the staging region, given the value type. s_p8,
// scores and s_alpha are sized for one head-pass (PF_PASS_HEADS) since the two
// passes reuse them sequentially.
template <typename T, int HEAD_DIM>
__host__ __device__ constexpr int prefill_smem_bytes() {
    (void)sizeof(T);
    // No sQ: the QK A-fragment is built per head-pass straight from L2 (Q is
    // invariant across key tiles), freeing 33.75KB so two blocks fit per SM.
    return PF_KEYS * (HEAD_DIM + 16)    // sK        int8 (padded for ldmatrix)
         + HEAD_DIM * PF_VPAD           // sVt       int8 (transposed, padded)
         + PF_PASS_HEADS * PF_KPAD      // s_p8      int8 (padded, per pass)
         + PF_PASS_HEADS * PF_SCR_LD * 4 // scores   f32 (band-collapsed, +1 word
                                          //          row stagger for the atomics)
         + PF_PASS_HEADS * NPAL * 4     // scaleQ    f32 (per pass — recomputed)
         + PF_KEYS * NPAL * 4           // scaleK    f32
         + PF_PASS_HEADS * 4            // s_alpha   f32 (per pass)
         + PF_KEYS * 4                  // key_valid int
         + HEAD_DIM * 4;                // s_vscale  f32 (per-dim V scale)
}

// Identity band layout only (dim d → band d/SUB). No palette-map routing.
template <typename T, int HEAD_DIM, int ROPE_DIM>
__global__ void __launch_bounds__(PF_WARPS * 32, 2)
latent_prefill_kernel(
    const T* __restrict__ q,               // [total_q, H, HEAD_DIM] pre-RoPE
    const uint8_t* __restrict__ headers,   // SlotHeader[1]
    const uint32_t* __restrict__ q_pos,    // [total_q]
    const T* __restrict__ kv_fresh,        // [fresh_rows, HEAD_DIM] pre-RoPE
    const int8_t* __restrict__ comp_i8,    // [G_total, HEAD_DIM] roped+per-band int8 (scratch)
    const float* __restrict__ comp_scale,  // [G_total, NPAL] per-band scale (scratch)
    const int8_t* __restrict__ comp_v8,    // [G_total, HEAD_DIM] per-dim-global int8 V (scratch)
    const float* __restrict__ comp_vmax,   // [HEAD_DIM] global per-dim max|v|
    const uint32_t* __restrict__ comp_idx, // [total_q, max_sel] ascending
    const uint32_t* __restrict__ comp_cnt, // [total_q]
    const uint32_t* __restrict__ comp_pos, // [G_total] entry position (causal guard)
    const float* __restrict__ rope_tab,    // factored cos/sin table (common)
    float* __restrict__ partial_acc,       // [total_q*H, splits, HEAD_DIM]
    float* __restrict__ partial_ml,        // [total_q*H, splits, 2]
    int total_q,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int fresh_rows,
    int fresh_base,
    int store_fmt   // writer-chunk float format tag (fresh-diagonal fake-quant)
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    constexpr int DPT = HEAD_DIM / 32;

    const int qi = (int)blockIdx.x;
    const int split_idx = (int)blockIdx.z;
    const int num_splits = (int)gridDim.z;
    const int tid = (int)threadIdx.x;
    const int warp = tid / 32;  // 0..15
    const int lane = tid % 32;
    if (qi >= total_q) return;

    // PV accumulator: warp = (row_tile, dim_group) over one head-pass. Each
    // n-slice holds 2 heads × 2 dims per lane; 8 n-slices span the group's 64
    // dims. Reused across both passes (emitted between passes).
    const int row_tile = warp >> 3;   // 0..1  → pass-heads [row_tile*16, +16)
    const int dim_group = warp & 7;   // 0..7  → dims  [dim_group*64, +64)
    float o_acc[PF_GDIMS / 8][4];

    // Softmax state: warp owns pass-heads {2w, 2w+1}, lane owns one key.
    float m_run[2];
    float l_run[2];

    const SlotHeader& slot = get_slot_header(headers, 0);
    const uint32_t n_slices = slot.n_slices;
    const uint64_t slices_ptr = slot.slices_ptr;
    const uint32_t n_sel = comp_cnt ? comp_cnt[qi] : 0;
    const int my_pos = (int)q_pos[qi];

    // Emit this pass's (m,l) — softmax warp owns pass-heads {2w,2w+1}.
    auto emit_ml = [&](int head_base) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int head = head_base + 2 * warp + h;
            if (head >= n_q_head || lane != 0) continue;
            int64_t base = ((int64_t)qi * n_q_head + head) * num_splits + split_idx;
            partial_ml[base * 2] = m_run[h];
            partial_ml[base * 2 + 1] = l_run[h];
        }
    };
    // Emit this pass's o_acc (ΣpV) — PV warp owns (row_tile, dim_group).
    auto emit_o = [&](int head_base) {
        const int r0 = lane >> 2;         // 0..7
        const int c0 = (lane & 3) * 2;    // 0,2,4,6
        #pragma unroll
        for (int s = 0; s < PF_GDIMS / 8; ++s) {
            const int dbase = dim_group * PF_GDIMS + s * 8;
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                int head = head_base + row_tile * 16 + r0 + rr * 8;
                if (head >= n_q_head) continue;
                int64_t base = ((int64_t)qi * n_q_head + head) * num_splits + split_idx;
                float* acc = partial_acc + base * HEAD_DIM;
                acc[dbase + c0] = o_acc[s][rr * 2];
                acc[dbase + c0 + 1] = o_acc[s][rr * 2 + 1];
            }
        }
    };

    if (n_slices == 0 && n_sel == 0 && fresh_rows == 0) {
        #pragma unroll
        for (int s = 0; s < PF_GDIMS / 8; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
        m_run[0] = m_run[1] = -1e38f;
        l_run[0] = l_run[1] = 0.f;
        #pragma unroll
        for (int hpass = 0; hpass < PF_PASSES; ++hpass) {
            emit_ml(hpass * PF_PASS_HEADS);
            emit_o(hpass * PF_PASS_HEADS);
        }
        return;
    }

    // ─── Dynamic shared memory partition ──────────────────────────────────
    extern __shared__ __align__(128) char smem_raw[];
    constexpr int QLD = HEAD_DIM + 16;  // padded head-dim stride for ldmatrix
    constexpr int OFF_SK = 0;  // sK first — no sQ (Q A-frag built from L2 per pass)
    constexpr int OFF_VT = OFF_SK + PF_KEYS * QLD;
    constexpr int OFF_P8 = OFF_VT + HEAD_DIM * PF_VPAD;
    constexpr int OFF_SCR = OFF_P8 + PF_PASS_HEADS * PF_KPAD;
    constexpr int OFF_SCQ = OFF_SCR + PF_PASS_HEADS * PF_SCR_LD * 4;
    constexpr int OFF_SCK = OFF_SCQ + PF_PASS_HEADS * NPAL * 4;
    constexpr int OFF_ALP = OFF_SCK + PF_KEYS * NPAL * 4;
    constexpr int OFF_VAL = OFF_ALP + PF_PASS_HEADS * 4;
    constexpr int OFF_VSC = OFF_VAL + PF_KEYS * 4;
    int8_t (*sK)[QLD] = reinterpret_cast<int8_t (*)[QLD]>(smem_raw + OFF_SK);
    int8_t (*sVt)[PF_VPAD] = reinterpret_cast<int8_t (*)[PF_VPAD]>(smem_raw + OFF_VT);
    int8_t (*s_p8)[PF_KPAD] = reinterpret_cast<int8_t (*)[PF_KPAD]>(smem_raw + OFF_P8);
    float (*scores)[PF_SCR_LD] = reinterpret_cast<float (*)[PF_SCR_LD]>(smem_raw + OFF_SCR);
    float (*scaleQ)[NPAL] = reinterpret_cast<float (*)[NPAL]>(smem_raw + OFF_SCQ);
    float (*scaleK)[NPAL] = reinterpret_cast<float (*)[NPAL]>(smem_raw + OFF_SCK);
    float* s_alpha = reinterpret_cast<float*>(smem_raw + OFF_ALP);
    int* key_valid = reinterpret_cast<int*>(smem_raw + OFF_VAL);
    float* s_vscale = reinterpret_cast<float*>(smem_raw + OFF_VSC);

    // scaleQ (per head-band max) is computed per pass inside the head-pass loop
    // (only the pass's 32 heads), so the buffer is PF_PASS_HEADS-wide — saving
    // the smem that lets 2 blocks fit per SM. Q values themselves are rebuilt
    // into the QK A-fragment per pass, direct from L2.

    // Window tiling (32-key tiles).
    auto slice_len_of = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, 1);
        return (int)slice_len(sl);
    };
    auto slice_tiles = [&](int s) -> int {
        return (slice_len_of(s) + PF_KEYS - 1) / PF_KEYS;
    };
    auto tile_to_slice = [&](int tile_idx, int& slice_out, int& within_base_out) {
        int base = 0, s = 0;
        while (s + 1 < (int)n_slices) {
            int st = slice_tiles(s);
            if (base + st <= tile_idx) { base += st; ++s; } else break;
        }
        slice_out = s;
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, 1);
        within_base_out = (int)slice_offset(sl) + (tile_idx - base) * PF_KEYS;
    };

    int n_win_tiles = 0;
    for (int s = 0; s < (int)n_slices; ++s) n_win_tiles += slice_tiles(s);
    const int n_fresh_tiles = (fresh_rows + PF_KEYS - 1) / PF_KEYS;
    const int n_comp_tiles = ((int)n_sel + PF_KEYS - 1) / PF_KEYS;
    const int n_tiles = n_win_tiles + n_fresh_tiles + n_comp_tiles;

    const int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
    int tile_lo = split_idx * tiles_per_split;
    int tile_hi = tile_lo + tiles_per_split;
    if (tile_lo > n_tiles) tile_lo = n_tiles;
    if (tile_hi > n_tiles) tile_hi = n_tiles;

    // Load key `key` of the tile → roped f32 regs. A 512-thread block owns 16
    // warps = 16 keys per round, so each warp stages two keys (`warp` and
    // `warp+16`) per tile. The long-latency global reads (load_raw_key) are
    // split from the RoPE+quant (stage_key) so a tile's two keys can overlap
    // the second key's load with the first key's compute.
    auto load_raw_key = [&](int tile_idx, int key, float regs[DPT], int& key_pos, bool& valid) {
        valid = false;
        key_pos = 0;
        if (tile_idx < n_win_tiles) {
            int sl_idx, within_base;
            tile_to_slice(tile_idx, sl_idx, within_base);
            int within = within_base + key;
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, sl_idx, 1);
            int off = (int)slice_offset(sl);
            if (sl_idx < (int)n_slices && within < off + slice_len_of(sl_idx)) {
                key_pos = (int)slice_rope(sl) + (within - off);
                if (key_pos <= my_pos && key_pos > my_pos - window_size) {
                    valid = true;
                    const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, 0);
                    // Identity layout: one band per lane (DPT | SUB) —
                    // resolve {ptr, fmt, outer} once, format-dispatch each
                    // element.
                    auto ident_read = [&] {
                        const int bnd = (lane * DPT) / SUB;
                        const uint64_t bp = kvhead_k_ptr<HEAD_DIM, NPAL>(head_ptr, bnd);
                        const int fmt = kvhead_k_fmt<HEAD_DIM, NPAL>(head_ptr, bnd);
                        const float outer = kvhead_k_scale<HEAD_DIM, NPAL>(head_ptr, bnd);
                        if (bp && fmt == ArenaFormat::F8E4M3) {
                            // Hot path (writer chunk + uncompressed sealed):
                            // the direct FP8 row read, format check hoisted.
                            const uint8_t* src = (const uint8_t*)(uintptr_t)bp;
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) {
                                int d = lane * DPT + j;
                                regs[j] =
                                    fp8_to_f32(src[(int64_t)within * SUB + (d % SUB)]) / outer;
                            }
                        } else if (bp) {
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) {
                                int d = lane * DPT + j;
                                regs[j] = load_band_elem<SUB>(bp, fmt, outer, within, d % SUB);
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
                        }
                    };
                    // Identity band layout is the only layout (no pal_map).
                    ident_read();
                }
            }
        } else if (tile_idx < n_win_tiles + n_fresh_tiles) {
            int fj = (tile_idx - n_win_tiles) * PF_KEYS + key;
            if (fj < fresh_rows) {
                key_pos = fresh_base + fj;
                if (key_pos <= my_pos && key_pos > my_pos - window_size) {
                    valid = true;
                    const T* src = kv_fresh + (int64_t)fj * HEAD_DIM;
                    // The fresh diagonal is attended before it is read back from
                    // the arena, so it must match the writer chunk's storage
                    // precision: FP8 storage fake-quants here; BF16/F16/F32
                    // storage is lossless for the bf16 source, so read direct.
                    const bool round_fp8 = (store_fmt == ArenaFormat::F8E4M3);
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        float v = to_f32<T>(src[lane * DPT + j]);
                        if (round_fp8) {
                            __nv_fp8_e4m3 enc = __nv_fp8_e4m3(v);
                            regs[j] = fp8_to_f32(*reinterpret_cast<uint8_t*>(&enc));
                        } else {
                            regs[j] = v;
                        }
                    }
                }
            }
        } else {
            int e = (tile_idx - n_win_tiles - n_fresh_tiles) * PF_KEYS + key;
            if (e < (int)n_sel) {
                uint32_t gid = comp_idx[(int64_t)qi * max_sel + e];
                // Causal guard: drop a compressed entry that sits in this query's
                // future (a selection/reassembly bug could place it there). The
                // softmax then masks the dropped key (key_valid == 0).
                if (gid != 0xFFFFFFFFu && (int)comp_pos[gid] <= my_pos) {
                    valid = true;
                    // Per-assembly BAKED-RoPE corpus scratch — NOT the decode's
                    // persistent position-free cache (`latent_quant_corpus_range_
                    // kernel`). Here `latent_rope_quant_corpus_kernel` already
                    // rotated each entry at its window position, and the shared
                    // int8-PV operand `comp_v8` is derived from these roped bytes,
                    // so this path must NOT rope again: `key_pos` stays 0 (set by
                    // load_raw_key), and `stage_key` rotates by identity (pos 0 →
                    // cos 1, sin 0). Do NOT set `key_pos = comp_pos[gid]` — that
                    // would silently double-rotate. The int8 load is 4× smaller
                    // than the old f32 comp read.
                    const int8_t* src = comp_i8 + (int64_t)gid * HEAD_DIM;
                    const float* scl = comp_scale + (int64_t)gid * NPAL;
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        int d = lane * DPT + j;
                        regs[j] = (float)src[d] * scl[d / SUB];
                    }
                }
            }
        }
        if (!valid) {
            #pragma unroll
            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
        }
    };
    // RoPE + per-band int8 quant of a raw key → sK / scaleK / key_valid.
    auto stage_key = [&](int key, const float regs_in[DPT], int key_pos, bool valid) {
        float regs[DPT];
        #pragma unroll
        for (int j = 0; j < DPT; ++j) regs[j] = regs_in[j];
        if (valid) {
            #pragma unroll
            for (int j = 0; j < DPT; j += 2) {
                int d = lane * DPT + j;
                if (d >= NOPE_DIM)
                    rope_pair<ROPE_DIM / 2>(regs[j], regs[j + 1], rope_tab, key_pos, (d - NOPE_DIM) >> 1);
            }
        }
        // A K lane owns DPT dims; a band spans SUB dims = SUB/DPT adjacent
        // lanes, so the reduction width and write predicate scale with it.
        constexpr int KLB = SUB / DPT;  // lanes per band
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < DPT; ++j) mx = fmaxf(mx, fabsf(regs[j]));
        #pragma unroll
        for (int off = 1; off < KLB; off <<= 1)
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        // Explicit reciprocal multiply (NOT `/ 127.f`): nvcc lowers constant
        // division to this multiply regardless of -prec-div, so the mirror
        // contract writes the op both sides compute.
        float sk = __fdiv_rn(mx, 127.f); // IEEE division (mirror parity)
        if (sk == 0.f) sk = 1.f;
        if ((lane & (KLB - 1)) == 0) scaleK[key][lane / KLB] = sk;
        float inv = __frcp_rn(sk); // IEEE reciprocal (mirror parity)
        // Pack the lane's 16 quantized bytes in registers and write the row
        // segment as ONE 16-byte store (16 byte-stores would each pay the same
        // 4-way bank phase — one wide store pays it once).
        uint32_t pk[DPT / 4];
        #pragma unroll
        for (int j = 0; j < DPT; ++j) {
            float v = fminf(fmaxf(regs[j] * inv, -127.f), 127.f);
            uint32_t b = (uint32_t)(uint8_t)(int8_t)__float2int_rn(v);
            if ((j & 3) == 0) pk[j / 4] = b;
            else pk[j / 4] |= b << (8 * (j & 3));
        }
        *reinterpret_cast<int4*>(&sK[key][lane * DPT]) =
            make_int4((int)pk[0], (int)pk[1], (int)pk[2], (int)pk[3]);
        if (lane == 0) key_valid[key] = valid ? 1 : 0;
    };
    // Stage a full 32-key tile into sK. Each warp loads then stages keys
    // {warp, warp+16}; the second key's global load overlaps the first key's
    // RoPE+quant.
    auto stage_tile = [&](int tile_idx) {
        float raw0[DPT], raw1[DPT];
        int kp0, kp1;
        bool v0, v1;
        load_raw_key(tile_idx, warp, raw0, kp0, v0);
        load_raw_key(tile_idx, warp + PF_WARPS, raw1, kp1, v1);
        stage_key(warp, raw0, kp0, v0);
        stage_key(warp + PF_WARPS, raw1, kp1, v1);
    };

    // Build the 4 packed int8 (one uint32) of Q for `head` over 4 consecutive
    // dims from `d0` (4-aligned): read bf16 from L2, RoPE at my_pos, quantize by
    // `inv` (= 1/scaleQ[head][band]), little-endian pack. Byte-for-byte matches
    // load_a_frag_m16k32's layout, so the hand-built A-fragment equals the old
    // ldmatrix-from-sQ result — but with no sQ smem (Q comes straight from L2).
    auto q_pack4 = [&](int head, int d0, float inv, bool ok) -> uint32_t {
        if (!ok) return 0u;
        const T* qp = q + ((int64_t)qi * n_q_head + head) * HEAD_DIM + d0;
        float v0 = to_f32<T>(qp[0]), v1 = to_f32<T>(qp[1]);
        float v2 = to_f32<T>(qp[2]), v3 = to_f32<T>(qp[3]);
        if (d0 >= NOPE_DIM) {
            rope_pair<ROPE_DIM / 2>(v0, v1, rope_tab, my_pos, (d0 - NOPE_DIM) >> 1);
            rope_pair<ROPE_DIM / 2>(v2, v3, rope_tab, my_pos, (d0 + 2 - NOPE_DIM) >> 1);
        }
        int i0 = __float2int_rn(fminf(fmaxf(v0 * inv, -127.f), 127.f));
        int i1 = __float2int_rn(fminf(fmaxf(v1 * inv, -127.f), 127.f));
        int i2 = __float2int_rn(fminf(fmaxf(v2 * inv, -127.f), 127.f));
        int i3 = __float2int_rn(fminf(fmaxf(v3 * inv, -127.f), 127.f));
        return ((uint32_t)(uint8_t)(int8_t)i0)
             | ((uint32_t)(uint8_t)(int8_t)i1 << 8)
             | ((uint32_t)(uint8_t)(int8_t)i2 << 16)
             | ((uint32_t)(uint8_t)(int8_t)i3 << 24);
    };

    // Two sequential head-passes of 32 heads each. sK/sVt/scores/s_p8 are
    // rebuilt per pass; o_acc/m_run/l_run are reused (emitted between passes).
    #pragma unroll 1
    for (int hpass = 0; hpass < PF_PASSES; ++hpass) {
        const int head_base = hpass * PF_PASS_HEADS;
        #pragma unroll
        for (int s = 0; s < PF_GDIMS / 8; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
        m_run[0] = m_run[1] = -1e38f;
        l_run[0] = l_run[1] = 0.f;

        // Per-pass scaleQ: this pass's 32 heads (pass-local index hl = tid/16),
        // RoPE at my_pos, per-band max. Recomputed each pass so the buffer is
        // only PF_PASS_HEADS wide (the smem that lets 2 blocks fit per SM).
        {
            const int hl = tid / 16;               // 0..31 pass-local head
            const int head = head_base + hl;
            const int dseg = (tid % 16) * 32;
            float qr[32];
            if (head < n_q_head) {
                const T* qp = q + ((int64_t)qi * n_q_head + head) * HEAD_DIM + dseg;
                #pragma unroll
                for (int j = 0; j < 32; ++j) qr[j] = to_f32<T>(qp[j]);
            } else {
                #pragma unroll
                for (int j = 0; j < 32; ++j) qr[j] = 0.f;
            }
            #pragma unroll
            for (int j = 0; j < 32; j += 2) {
                int d = dseg + j;
                if (d >= NOPE_DIM)
                    rope_pair<ROPE_DIM / 2>(qr[j], qr[j + 1], rope_tab, my_pos, (d - NOPE_DIM) >> 1);
            }
            // Each Q thread owns 32 dims; a band spans SUB dims = SUB/32
            // adjacent threads, so the reduction and predicate scale with it.
            constexpr int Q_THREADS_PER_BAND = SUB / 32;
            float mx = 0.f;
            #pragma unroll
            for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(qr[j]));
            #pragma unroll
            for (int off = 1; off < Q_THREADS_PER_BAND; off <<= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            float sq = __fdiv_rn(mx, 127.f); // IEEE division (see sk)
            if (sq == 0.f) sq = 1.f;
            if ((tid & (Q_THREADS_PER_BAND - 1)) == 0)
                scaleQ[hl][(tid % 16) / Q_THREADS_PER_BAND] = sq;
        }
        __syncthreads();

        // QK A-fragment for this pass's 32 heads — built ONCE from L2 and reused
        // across every key tile (Q is tile-invariant). Warp = (hgroup, band);
        // lane t holds rows {gbase+t>>2, +8} at dims per the m16n8k32 A layout.
        // Warp = band (0..NPAL-1); each warp builds the Q A-fragment for BOTH
        // row-tiles of its band (PF_WARPS=NPAL warps cover the 16 bands). At the
        // old NPAL=8 this was 16 warps = 2 row-tiles × 8 bands; at NPAL=16 the
        // band count fills the warps, so the two row-tiles loop per warp.
        constexpr int NKS = SUB / 32;
        uint32_t qa_frag[PF_ROW_TILES][NKS][4];
        if (warp < NPAL) {
            const int p = warp;
            const int rb = lane >> 2;
            const int cb = (lane & 3) * 4;
            #pragma unroll
            for (int rt = 0; rt < PF_ROW_TILES; ++rt) {
                const int gbase = head_base + rt * 16;
                const int hA = gbase + rb, hB = gbase + rb + 8;
                const bool okA = hA < n_q_head, okB = hB < n_q_head;
                const float invA = okA ? __frcp_rn(scaleQ[hA - head_base][p]) : 0.f;
                const float invB = okB ? __frcp_rn(scaleQ[hB - head_base][p]) : 0.f;
                #pragma unroll
                for (int ks = 0; ks < NKS; ++ks) {
                    const int d0 = p * SUB + ks * 32 + cb;
                    qa_frag[rt][ks][0] = q_pack4(hA, d0, invA, okA);
                    qa_frag[rt][ks][1] = q_pack4(hB, d0, invB, okB);
                    qa_frag[rt][ks][2] = q_pack4(hA, d0 + 16, invA, okA);
                    qa_frag[rt][ks][3] = q_pack4(hB, d0 + 16, invB, okB);
                }
            }
        }

        for (int tile = tile_lo; tile < tile_hi; ++tile) {
            stage_tile(tile);
            const bool comp_tile = tile >= n_win_tiles + n_fresh_tiles;
            if (comp_tile) {
                // Comp tile: sVt is a pure byte gather of the pre-quantized
                // per-dim-global operand (comp_v8) — no per-tile max/requant,
                // no sK re-read. Runs before the barrier: the previous tile's
                // PV finished at the loop-end barrier, and nothing here reads
                // sK, so the gather's global loads overlap the sK staging.
                // Lane l holds key l's gid; a warp shuffle broadcasts it.
                const int e0 = (tile - n_win_tiles - n_fresh_tiles) * PF_KEYS;
                const uint32_t my_gid = (e0 + lane < (int)n_sel)
                    ? comp_idx[(int64_t)qi * max_sel + e0 + lane]
                    : 0xFFFFFFFFu;
                // The epilogue scale is the kernel-constant comp_vmax; comp
                // tiles are the contiguous tail, so fill s_vscale once at the
                // split's first comp tile (window tiles rewrite it per tile).
                const int comp_lo = n_win_tiles + n_fresh_tiles;
                const int first_comp = tile_lo > comp_lo ? tile_lo : comp_lo;
                for (int d = tid; d < HEAD_DIM; d += PF_WARPS * 32) {
                    if (tile == first_comp) s_vscale[d] = comp_vmax[d];
                    // Row d of sVt is the tile's 32 keys as 32 CONSECUTIVE
                    // bytes: pack each gathered byte into words and write the
                    // row as two 16-byte stores (the unrolled byte loads are
                    // independent, so they issue together and overlap).
                    uint32_t packed[PF_KEYS / 4];
                    #pragma unroll
                    for (int k4 = 0; k4 < PF_KEYS / 4; ++k4) {
                        uint32_t w = 0;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            uint32_t g = __shfl_sync(0xffffffffu, my_gid, k4 * 4 + j);
                            int8_t b = (g != 0xFFFFFFFFu)
                                ? comp_v8[(int64_t)g * HEAD_DIM + d]
                                : (int8_t)0;
                            w |= (uint32_t)(uint8_t)b << (8 * j);
                        }
                        packed[k4] = w;
                    }
                    int4* dst = reinterpret_cast<int4*>(&sVt[d][0]);
                    dst[0] = make_int4((int)packed[0], (int)packed[1], (int)packed[2], (int)packed[3]);
                    dst[1] = make_int4((int)packed[4], (int)packed[5], (int)packed[6], (int)packed[7]);
                }
            }
            // Zero the band-collapsed scores for this pass's tile (full padded
            // extent — flat indexing walks the staggered rows).
            #pragma unroll
            for (int c = tid; c < PF_PASS_HEADS * PF_SCR_LD; c += PF_WARPS * 32)
                reinterpret_cast<float*>(scores)[c] = 0.f;
            __syncthreads();

            if (!comp_tile) {
                // Window/fresh tile: build sVt[dim][key] from sK with a
                // PER-DIM scale — max|v| over the tile's keys for that output
                // dim. The scale is constant across keys (the PV contraction
                // index), so it factors out of the MMA; per-dim keeps int8
                // precision on small dims. One thread per dim reduces its 32
                // keys, then requantizes them. (Reads sK → must follow the
                // barrier; comp tiles skip this second barrier entirely.)
                for (int d = tid; d < HEAD_DIM; d += PF_WARPS * 32) {
                    int bnd = d / SUB;
                    float vmax = 0.f;
                    #pragma unroll
                    for (int k = 0; k < PF_KEYS; ++k)
                        vmax = fmaxf(vmax, fabsf((float)sK[k][d] * scaleK[k][bnd]));
                    s_vscale[d] = vmax;
                    float inv = (vmax > 0.f) ? (127.f / vmax) : 0.f;
                    #pragma unroll
                    for (int k = 0; k < PF_KEYS; ++k) {
                        float v = (float)sK[k][d] * scaleK[k][bnd] * inv; // v·127/vmax
                        sVt[d][k] = (int8_t)fminf(fmaxf(rintf(v), -127.f), 127.f);
                    }
                }
                __syncthreads();
            }

            // QK: warp = band; each warp does BOTH row-tiles of the pass's 32
            // heads. m16n8k32 per band; the band's scaled score is atomic-summed
            // into scores[pass-head][key] (band-collapsed over the NPAL bands).
            if (warp < NPAL) {
                const int p = warp;
                const int r0 = lane >> 2;
                const int c0 = (lane & 3) * 2;
                // qa_frag (this pass's Q, built from L2 above) is reused across
                // every n-group and every tile — no per-tile ldmatrix, no sQ.
                #pragma unroll
                for (int rt = 0; rt < PF_ROW_TILES; ++rt) {
                    const int hbase = rt * 16;             // pass-local head base
                    #pragma unroll
                    for (int ng = 0; ng < PF_KEYS / 8; ++ng) {
                        const int kb = ng * 8;
                        int32_t c[4] = {0, 0, 0, 0};
                        uint32_t b_cur[2], b_nxt[2];
                        fused_attn::load_b_frag_n8k32_ldmatrix(b_cur, &sK[kb][p * SUB], QLD, lane);
                        #pragma unroll
                        for (int ks = 0; ks < NKS; ++ks) {
                            if (ks + 1 < NKS)
                                fused_attn::load_b_frag_n8k32_ldmatrix(b_nxt, &sK[kb][p * SUB + (ks + 1) * 32], QLD, lane);
                            fused_attn::mma_int8_m16n8k32(c, qa_frag[rt][ks], b_cur, c);
                            b_cur[0] = b_nxt[0]; b_cur[1] = b_nxt[1];
                        }
                        float qA = scaleQ[hbase + r0][p], qB = scaleQ[hbase + r0 + 8][p];
                        atomicAdd(&scores[hbase + r0][kb + c0], (float)c[0] * qA * scaleK[kb + c0][p]);
                        atomicAdd(&scores[hbase + r0][kb + c0 + 1], (float)c[1] * qA * scaleK[kb + c0 + 1][p]);
                        atomicAdd(&scores[hbase + r0 + 8][kb + c0], (float)c[2] * qB * scaleK[kb + c0][p]);
                        atomicAdd(&scores[hbase + r0 + 8][kb + c0 + 1], (float)c[3] * qB * scaleK[kb + c0 + 1][p]);
                    }
                }
            }
            __syncthreads();

            // Softmax: warp owns pass-heads {2w,2w+1}; lane owns key `lane`.
            // Online m/l; P=exp(sc-m) → int8 (×127) in s_p8; alpha → smem.
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int lh = 2 * warp + h;               // pass-local head 0..31
                const bool live = head_base + lh < n_q_head;
                float sc = (live && key_valid[lane]) ? scores[lh][lane] * softmax_scale : -1e38f;
                float m_tile = sc;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, o));
                float m_new = fmaxf(m_run[h], m_tile);
                float alpha = (m_run[h] <= -1e37f) ? 0.f : ds_exp(m_run[h] - m_new);
                float p = (sc <= -1e37f || m_new <= -1e37f) ? 0.f : ds_exp(sc - m_new);
                if (live) s_p8[lh][lane] = (int8_t)__float2int_rn(fminf(127.f, p * 127.f));
                float l_add = p;
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) l_add += __shfl_xor_sync(0xffffffff, l_add, o);
                l_run[h] = l_run[h] * alpha + l_add;
                m_run[h] = m_new;
                if (live && lane == 0) s_alpha[lh] = alpha;
            }
            __syncthreads();

            // PV: warp = (row_tile, dim_group). P is s_p8[row_tile*16][*] (16
            // pass-heads × 32 keys); each 8-dim n-slice is one m16n8k32 against
            // sVt[dim_group*64 + s*8][*] (8 dims × 32 keys). Rescale o_acc by the
            // per-head alpha, then add this tile's contribution.
            {
                uint32_t pa[4];
                fused_attn::load_a_frag_m16k32_ldmatrix(pa, &s_p8[row_tile * 16][0], PF_KPAD, lane);
                const float a0 = s_alpha[row_tile * 16 + (lane >> 2)];
                const float a1 = s_alpha[row_tile * 16 + (lane >> 2) + 8];
                const int c0 = (lane & 3) * 2;
                // sVt ≈ v·127/vmax_dim and P_int8 ≈ p·127 → ΣpV = d_i·vmax_dim/127².
                constexpr float inv127sq = 1.f / (127.f * 127.f);
                // Software-pipelined: slice s+1's Vᵀ fragment issues (ldmatrix)
                // before slice s's MMA, so the smem-load latency overlaps the
                // tensor-core op instead of stalling on it.
                constexpr int NS = PF_GDIMS / 8;
                const int dbase0 = dim_group * PF_GDIMS;
                uint32_t vb_cur[2], vb_nxt[2];
                fused_attn::load_b_frag_n8k32_ldmatrix(vb_cur, &sVt[dbase0][0], PF_VPAD, lane);
                #pragma unroll
                for (int s = 0; s < NS; ++s) {
                    const int dbase = dbase0 + s * 8;
                    if (s + 1 < NS)
                        fused_attn::load_b_frag_n8k32_ldmatrix(vb_nxt, &sVt[dbase + 8][0], PF_VPAD, lane);
                    int32_t d_i[4], c0i[4] = {0, 0, 0, 0};
                    fused_attn::mma_int8_m16n8k32(d_i, pa, vb_cur, c0i);
                    float vs0 = s_vscale[dbase + c0] * inv127sq;
                    float vs1 = s_vscale[dbase + c0 + 1] * inv127sq;
                    o_acc[s][0] = o_acc[s][0] * a0 + (float)d_i[0] * vs0;
                    o_acc[s][1] = o_acc[s][1] * a0 + (float)d_i[1] * vs1;
                    o_acc[s][2] = o_acc[s][2] * a1 + (float)d_i[2] * vs0;
                    o_acc[s][3] = o_acc[s][3] * a1 + (float)d_i[3] * vs1;
                    vb_cur[0] = vb_nxt[0];
                    vb_cur[1] = vb_nxt[1];
                }
            }
            __syncthreads();
        }

        emit_ml(head_base);
        emit_o(head_base);
    }
}

// Prefill combine: as the decode combine, but the query position comes from
// the per-query array (no writer-slice derivation).
template <typename O, int HEAD_DIM, int ROPE_DIM>
__global__ void latent_prefill_combine_kernel(
    O* __restrict__ out,                    // [total_q, H, HEAD_DIM]
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_ml,
    const uint32_t* __restrict__ q_pos,     // [total_q]
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
    const int qi = row / n_q_head;
    const int head = row % n_q_head;

    const float* ml = partial_ml + (int64_t)row * num_splits * 2;
    const float* pa = partial_acc + (int64_t)row * num_splits * HEAD_DIM;

    float gm = -1e38f;
    for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, ml[s * 2]);
    const float sink = sinks[head];
    const float m_fin = fmaxf(gm, sink);

    float acc = 0.f, L = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = ml[s * 2];
        if (!(m_s > -1e37f)) continue;
        float w = ds_exp(m_s - m_fin);
        acc = __fmaf_rn(pa[(int64_t)s * HEAD_DIM + d], w, acc);
        L = __fmaf_rn(ml[s * 2 + 1], w, L);
    }
    L += ds_exp(sink - m_fin);
    float val = acc / fmaxf(L, 1e-10f);

    if (d >= NOPE_DIM) {
        float partner = __shfl_xor_sync(0xffffffff, val, 1);
        float c, s;
        rope_lookup<ROPE_DIM / 2>(rope_tab, (int)q_pos[qi], (d - NOPE_DIM) >> 1, s, c);
        val = (d & 1) == 0
            ? __fadd_rn(__fmul_rn(val, c), __fmul_rn(partner, s))
            : __fsub_rn(__fmul_rn(val, c), __fmul_rn(partner, s));
    }
    out[(int64_t)row * HEAD_DIM + d] = from_f32<O>(val);
}

// Per-prefill corpus pre-RoPE + per-band int8 quant, plus the per-dim |v|
// maxima that scale the pre-quantized PV operand.
//
// This scratch is BAKED-RoPE and per-assembly — a DIFFERENT object from the
// decode path's persistent POSITION-FREE cache (`latent_quant_corpus_range_
// kernel`), even though both are named `comp_i8`. Prefill can't share the
// position-free contract because its int8-PV operand `comp_v8` is derived from
// these bytes and gathered straight into the PV matmul with no rotation on that
// path — making the K bytes position-free would leave V un-rotated while K is
// rotated. So this bakes the rotation in and is rebuilt whenever positions
// change; the attention kernel reads it with `key_pos == 0` (identity rotation).
//
// `comp`, `comp_pos` and `comp_idx` are all the caller-compacted SELECTED UNION
// — the host gathers exactly the distinct entries some query in this prefill
// attends (deduped) and remaps `comp_idx` into it; `comp_pos` is index-aligned to
// the SAME compacted gid used here and by the attention causal guard, so passing
// an un-compacted `comp_pos` is silently wrong. `g_total` is thus the attended
// set, not the whole gallery: the pre-pass is O(attended), and `comp_vmax` is the
// per-dim max over exactly that attended union (bounded, no whole-corpus
// inflation; still `max_sel × total_q` wide worst case — a percentile clamp would
// bound it further). Rope each entry at its group-start position and quantize per
// band. Grid-stride over entries; block = NPAL warps (one per band), so each
// thread's dims are fixed and the per-dim max accumulates in registers, hitting
// global memory with ONE atomicMax per (thread, dim) at the end. The canonical
// gallery (`comp`) stays f32/pre-RoPE and position-free (§C).
template <int HEAD_DIM, int ROPE_DIM>
__global__ void latent_rope_quant_corpus_kernel(
    const int8_t* __restrict__ nope_i8,    // [G, NOPE_DIM] two-region: nope int8
    const float* __restrict__ nope_scale,  // [G, NOPE_BANDS] per-nope-band scale
    const __nv_bfloat16* __restrict__ rope_bf, // [G, ROPE_DIM] rope pre-rotation bf16
    const uint32_t* __restrict__ comp_pos, // [G]
    const float* __restrict__ rope_tab,
    int8_t* __restrict__ comp_i8,          // [G, HEAD_DIM] baked int8 (out)
    float* __restrict__ comp_scale,        // [G, NPAL] per-band scale (out)
    float* __restrict__ comp_vmax,         // [HEAD_DIM] global per-dim max|v| (ZEROED in)
    int g_total
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    constexpr int NOPE_BANDS = NOPE_DIM / SUB;
    constexpr int DPL = SUB / 32;  // dims per lane within a band
    // The rope/nope split MUST fall on a band boundary. Below, the DPL==1 rope
    // branch is `if (d >= NOPE_DIM)` — a PER-DIM predicate — followed by a
    // full-mask `__shfl_xor_sync`. It is warp-uniform (all lanes of a warp take
    // the branch together) only when NOPE_DIM is a multiple of SUB, so no band
    // straddles the boundary. Off-boundary → a divergent full-mask shuffle, which
    // is undefined (silent wrong result or hang), not a compile error.
    static_assert(NOPE_DIM % SUB == 0, "rope/nope split must fall on a band boundary");
    const int band = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x % 32;
    float dmax[DPL];
    #pragma unroll
    for (int j = 0; j < DPL; ++j) dmax[j] = 0.f;
    for (int gid = (int)blockIdx.x; gid < g_total; gid += (int)gridDim.x) {
        const int pos = (int)comp_pos[gid];
        int8_t* dst = comp_i8 + (int64_t)gid * HEAD_DIM;
        // Read the SAME two-region cache the decode reads: nope int8·scale, rope
        // bf16 pre-rotation. Both paths thus derive from one representation — the
        // prefill's baked int8 differs from decode only by the extra bake round.
        const int8_t* nsrc = nope_i8 + (int64_t)gid * NOPE_DIM;
        const float* nscl = nope_scale + (int64_t)gid * NOPE_BANDS;
        const __nv_bfloat16* rsrc = rope_bf + (int64_t)gid * ROPE_DIM;
        float v[DPL];
        #pragma unroll
        for (int j = 0; j < DPL; ++j) {
            int d = band * SUB + lane * DPL + j;
            v[j] = (d < NOPE_DIM)
                ? (float)nsrc[d] * nscl[d / SUB]
                : __bfloat162float(rsrc[d - NOPE_DIM]);
        }
        // RoPE — register-local pairs when DPL≥2; cross-lane partner shuffle
        // when SUB=32 (DPL=1). See the range kernel for the parity argument.
        if constexpr (DPL >= 2) {
            #pragma unroll
            for (int j = 0; j < DPL; j += 2) {
                int d = band * SUB + lane * DPL + j;
                if (d >= NOPE_DIM)
                    rope_pair<ROPE_DIM / 2>(v[j], v[j + 1], rope_tab, pos, (d - NOPE_DIM) >> 1);
            }
        } else {
            int d = band * SUB + lane;
            if (d >= NOPE_DIM) {
                float partner = __shfl_xor_sync(0xffffffff, v[0], 1);
                bool even = (d & 1) == 0;
                float x0 = even ? v[0] : partner;
                float x1 = even ? partner : v[0];
                rope_pair<ROPE_DIM / 2>(x0, x1, rope_tab, pos, (d - NOPE_DIM) >> 1);
                v[0] = even ? x0 : x1;
            }
        }
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < DPL; ++j) {
            float a = fabsf(v[j]);
            mx = fmaxf(mx, a);
            dmax[j] = fmaxf(dmax[j], a);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float scale = __fdiv_rn(mx, 127.f); // IEEE division (see sk)
        if (scale == 0.f) scale = 1.f;
        if (lane == 0) comp_scale[(int64_t)gid * NPAL + band] = scale;
        float inv = __frcp_rn(scale); // IEEE reciprocal (mirror parity)
        #pragma unroll
        for (int j = 0; j < DPL; ++j) {
            int d = band * SUB + lane * DPL + j;
            float qv = fminf(fmaxf(v[j] * inv, -127.f), 127.f);
            dst[d] = (int8_t)__float2int_rn(qv);
        }
    }
    // Fold this thread's register maxima into the global per-dim maxima.
    // Values are non-negative, so IEEE floats order like their int bits.
    #pragma unroll
    for (int j = 0; j < DPL; ++j) {
        int d = band * SUB + lane * DPL + j;
        atomicMax(reinterpret_cast<int*>(&comp_vmax[d]), __float_as_int(dmax[j]));
    }
}

// Pre-quantized PV operand: comp_v8[g][d] = round(127·v/comp_vmax[d]) where
// v = comp_i8·comp_scale is EXACTLY the value the attention kernel stages into
// sK for this entry (identity round-trip), so the only numerical delta vs the
// in-kernel sVt build is the per-dim scale being corpus-global instead of
// per-tile. With that scale constant, a comp tile's transposed V operand
// becomes a pure byte gather — no per-tile max/requant — and the PV epilogue
// scale is a per-kernel constant.
template <int HEAD_DIM>
__global__ void latent_quant_v_corpus_kernel(
    const int8_t* __restrict__ comp_i8,    // [G, HEAD_DIM] roped+per-band int8
    const float* __restrict__ comp_scale,  // [G, NPAL]
    const float* __restrict__ comp_vmax,   // [HEAD_DIM] global per-dim max|v|
    int8_t* __restrict__ comp_v8,          // [G, HEAD_DIM] per-dim-global int8 (out)
    int g_total
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    const int64_t n = (int64_t)g_total * HEAD_DIM;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        const int d = (int)(i % HEAD_DIM);
        const int64_t g = i / HEAD_DIM;
        const float v = (float)comp_i8[i] * comp_scale[g * NPAL + d / SUB];
        const float vm = comp_vmax[d];
        const float qv = vm > 0.f ? rintf(v * 127.f / vm) : 0.f;
        comp_v8[i] = (int8_t)fminf(fmaxf(qv, -127.f), 127.f);
    }
}

template <typename T, int HEAD_DIM, int ROPE_DIM>
void launch_latent_prefill(
    const T* q,
    const uint8_t* headers,
    float* out, // final attention output is F32 (fed straight to int8 out-proj)
    const uint32_t* q_pos,
    const T* kv_fresh,
    // Two-region corpus cache (the same the decode reads): nope int8 [G,NOPE_DIM]
    // + per-nope-band scale [G,NOPE_BANDS], rope pre-rotation bf16 [G,ROPE_DIM].
    const int8_t* nope_i8,
    const float* nope_scale,
    const __nv_bfloat16* rope_bf,
    const uint32_t* comp_pos,
    const uint32_t* comp_idx,
    const uint32_t* comp_cnt,
    const float* sinks,
    const float* rope_tab,
    float* pa,
    float* pm,
    int8_t* comp_i8,       // per-prefill roped+quant corpus scratch [G, HEAD_DIM]
    float* comp_scale,     // per-band scale [G, NPAL]
    int8_t* comp_v8,       // per-dim-global int8 V scratch [G, HEAD_DIM]
    float* comp_vmax,      // global per-dim max|v| [HEAD_DIM] (ZEROED by caller)
    int g_total,
    int total_q,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int fresh_rows,
    int fresh_base,
    int num_splits,
    // Writer-chunk float format tag: the fresh diagonal fake-quants to it.
    int store_fmt,
    cudaStream_t stream
) {
    if (total_q <= 0 || n_q_head <= 0 || num_splits < 1) return;
    if (n_q_head > PF_HEADS) return;

    // Rope + per-band int8-quantize the corpus once (positions fixed for the
    // prefill), fold the global per-dim |v| maxima, then quantize the PV
    // operand against them; the attention kernel then reads the cached int8
    // (QK) and gathers the pre-quantized V bytes (PV), skipping the per-query
    // RoPE and the per-tile V requant. Runs on the first chunk (g_total > 0);
    // later chunks share the scratch on the ordered stream.
    if (g_total > 0) {
        const int pre_blocks = g_total < 2048 ? g_total : 2048;
        latent_rope_quant_corpus_kernel<HEAD_DIM, ROPE_DIM>
            <<<pre_blocks, NPAL * 32, 0, stream>>>(
                nope_i8, nope_scale, rope_bf, comp_pos, rope_tab, comp_i8,
                comp_scale, comp_vmax, g_total);
        const int64_t n_elem = (int64_t)g_total * HEAD_DIM;
        int v8_blocks = (int)((n_elem + 255) / 256);
        if (v8_blocks > 65535) v8_blocks = 65535;
        latent_quant_v_corpus_kernel<HEAD_DIM><<<v8_blocks, 256, 0, stream>>>(
            comp_i8, comp_scale, comp_vmax, comp_v8, g_total);
    }

    constexpr int smem = prefill_smem_bytes<T, HEAD_DIM>();
    static bool attr_set = false;
    if (!attr_set) {
        const void* fn = (const void*)latent_prefill_kernel<T, HEAD_DIM, ROPE_DIM>;
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(
            fn, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        attr_set = true;
    }

    dim3 grid(total_q, 1, num_splits);
    dim3 block(PF_WARPS * 32);
    latent_prefill_kernel<T, HEAD_DIM, ROPE_DIM><<<grid, block, smem, stream>>>(
        q, headers, q_pos, kv_fresh, comp_i8, comp_scale, comp_v8, comp_vmax,
        comp_idx, comp_cnt, comp_pos, rope_tab, pa, pm, total_q, n_q_head, softmax_scale,
        window_size, max_sel, fresh_rows, fresh_base, store_fmt);

    const int num_rows = total_q * n_q_head;
    latent_prefill_combine_kernel<float, HEAD_DIM, ROPE_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
        out, pa, pm, q_pos, sinks, rope_tab, num_rows, n_q_head, num_splits);
}
}  // namespace latent_attn
