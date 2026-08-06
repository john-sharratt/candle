#pragma once
// =============================================================================
// deepseek_decode_kernel.cuh — batched/paged hybrid decode attention for
// DeepSeek-V4-Flash (single-latent K≡V, MQA n_kv_head=1).
//
// Fork of the paged-decode INT8 kernel restructured for HEAD_DIM=512:
//   • K and V are the SAME 512-d latent: one arena read serves both the int8
//     QK^T and the FP PV accumulate (halves KV traffic; one smem region).
//   • M-tile = 16 query heads staged int8 in smem (all heads share the one
//     latent), N-tile = 8 keys, K = 512 = 4 palettes × 4 m16n8k32 k-steps.
//   • Hybrid two-source key stream feeding ONE online softmax: the sliding
//     window (arena slot walk, position-clamped to the last `window_size`
//     tokens) followed by the selected compressed entries (index-driven walk
//     over a device GID list).
//   • RoPE is computed IN-KERNEL from the YaRN-adjusted inverse frequencies
//     (ROPE_DIM/2 floats) — no position-indexed cos/sin table exists anywhere
//     on this path (a 1M-position table would be hundreds of MB). Only the
//     trailing ROPE_DIM dims rotate (nope‖rope split), interleaved pairs.
//   • The window arena is uniform FP8 E4M3 in ascending-dim band order
//     (4 bands of 128), so keys load with direct pointer math — no palette
//     map, no format dispatch. Band pointers/outer-scales still come from the
//     KvHead record (k_* fields; v_* ignored — K≡V).
//   • The kernel ALWAYS emits split-KV partials (un-normalized ΣwV, m, l in
//     the natural-e domain). The companion combine kernel merges splits,
//     folds the per-head learned sink, normalizes, DE-ROTATES the output's
//     rope dims at the query position (inverse rotation — linear, so it
//     commutes with the merge), and writes the final output.
//
// All exponentials use fast_exp's cubic-polynomial e^x (plain f32 arithmetic,
// reproducible bit-for-bit by the CPU mirror oracle).
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

#include "../fast_exp.cuh"
#include "../blocks.cuh"
#include "../mma/mma_wrappers.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../paged-decode/decode_helpers.cuh"

namespace deepseek_attn {

constexpr int HEADS_TILE = 16;  // MMA M dimension: query heads per block
constexpr int KEYS_TILE = 8;    // MMA N dimension: keys per tile
constexpr int WARPS = 8;        // block = 256 threads
constexpr int NPAL = 4;         // 128-dim bands (SUB = HEAD_DIM / NPAL)

__device__ __forceinline__ float ds_exp(float x) {
    // fast_exp cubic e^x (Softmax mode: lower clamp only). Bit-reproducible.
    return fast_exp::exp<float, fast_exp::Softmax, fast_exp::High>(x);
}

__device__ __forceinline__ float fp8_to_f32(uint8_t b) {
    __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}

// ─── Table-free, bit-mirrorable RoPE trig ────────────────────────────────────
// The angle pos·freq is reduced in DOUBLE precision (an f32 product is
// unusable at depth — ulp(10⁶ rad) ≈ 0.06 rad) down to a quadrant residual
// r ∈ [-π/4, π/4] plus quadrant k. sin/cos then come from short minimax
// polynomials in PLAIN f32 arithmetic (the archive compiles `-fmad=false`), so
// every operation — reduction included — is exact-rounded and reproduced
// bit-for-bit by the CPU mirror oracle. No `__sincosf`, no position table.
// Every operation below uses the explicit round-to-nearest intrinsics
// (`__fmul_rn`/`__fadd_rn`/`__dmul_rn`/…): the compiler is NOT permitted to
// contract an explicit intrinsic into an fma, so the arithmetic here is
// exact-rounded IEEE regardless of `-fmad` — the property the CPU mirror's
// plain Rust ops reproduce bit-for-bit.
__device__ __forceinline__ void rope_angle(int pos, float freq, float& r, int& k) {
    double a = __dmul_rn((double)pos, (double)freq);
    double t = floor(__dmul_rn(a, 0.15915494309189535)); // ·1/2π
    a = __dsub_rn(a, __dmul_rn(t, 6.283185307179586));   // [0, 2π)
    double q = floor(__dadd_rn(__dmul_rn(a, 0.6366197723675814), 0.5)); // ·2/π
    r = (float)__dsub_rn(a, __dmul_rn(q, 1.5707963267948966));
    k = ((int)q) & 3;
}

__device__ __forceinline__ void ds_sincos(float r, int k, float& s, float& c) {
    float x2 = __fmul_rn(r, r);
    // sin on [-π/4, π/4] (cephes sinf coefficients).
    float sp_in = -1.9515295891e-4f;
    sp_in = __fadd_rn(__fmul_rn(sp_in, x2), 8.3321608736e-3f);
    sp_in = __fadd_rn(__fmul_rn(sp_in, x2), -1.6666654611e-1f);
    float rt = __fmul_rn(r, x2);
    float sp = __fadd_rn(r, __fmul_rn(rt, sp_in));
    // cos on [-π/4, π/4] (cephes cosf coefficients).
    float cp_in = 2.443315711809948e-5f;
    cp_in = __fadd_rn(__fmul_rn(cp_in, x2), -1.388731625493765e-3f);
    cp_in = __fadd_rn(__fmul_rn(cp_in, x2), 4.166664568298827e-2f);
    float x4 = __fmul_rn(x2, x2);
    float cp = __fsub_rn(1.0f, __fmul_rn(0.5f, x2));
    cp = __fadd_rn(cp, __fmul_rn(x4, cp_in));
    switch (k) {
        case 0: s = sp;  c = cp;  break;
        case 1: s = cp;  c = -sp; break;
        case 2: s = -sp; c = -cp; break;
        default: s = -cp; c = sp; break;
    }
}

// Interleaved-pair RoPE on a [pair0, pair1] register pair at position `pos`,
// frequency `freq`. Forward: (x0 c − x1 s, x0 s + x1 c) — explicit-rounded so
// the rotation cannot be contracted either.
__device__ __forceinline__ void rope_pair(float& x0, float& x1, int pos, float freq) {
    float r;
    int k;
    rope_angle(pos, freq, r, k);
    float c, s;
    ds_sincos(r, k, s, c);
    float r0 = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
    float r1 = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
    x0 = r0;
    x1 = r1;
}

// =============================================================================
// Decode kernel.
//
// grid  = (num_slots, head_tiles = ceil(H / 16), num_splits)
// block = 256 (8 warps)
//
// Thread roles:
//   Q stage    : thread t owns (head_local = t/16, 32 dims at (t%16)*32).
//   Key stage  : warp w owns key w of the tile; lane owns 16 dims (lane*16).
//   QK         : warps 0-3 own palette p = warp; 4 k-steps of m16n8k32 each,
//                int32-accumulated (one scale per palette), scaled into
//                scores_p[p][16][8]; the softmax owner sums the 4 palettes.
//   softmax/PV : warp w owns heads {2w, 2w+1}; lane owns 16 dims per head
//                (out accumulator = 32 f32/thread).
// =============================================================================
template <typename T, int HEAD_DIM, int ROPE_DIM>
__global__ void __launch_bounds__(WARPS * 32, 4)
deepseek_decode_kernel(
    const T* __restrict__ q,           // [slots, H, HEAD_DIM] pre-RoPE
    const uint8_t* __restrict__ headers,
    const T* __restrict__ kv_new,      // [slots, HEAD_DIM] pre-RoPE latent
    const float* __restrict__ comp,    // [G_total, HEAD_DIM] pre-RoPE entries
    const uint32_t* __restrict__ comp_pos, // [G_total] group-start positions
    const uint32_t* __restrict__ comp_idx, // [slots, max_sel] ascending GIDs
    const uint32_t* __restrict__ comp_cnt, // [slots]
    const float* __restrict__ rope_freqs,  // [ROPE_DIM/2]
    float* __restrict__ partial_acc,   // [slots*H, splits, HEAD_DIM]
    float* __restrict__ partial_ml,    // [slots*H, splits, 2]
    int num_slots,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    // Nullable stage-dump (slot 0 / head-tile 0 / split 0 / tile 0 only), for
    // the mirror oracle's stage-by-stage comparison:
    //   [0..64)          scaleQ[16][4]
    //   [64..8256)       sQ[16][512] (as float)
    //   [8256..8288)     scaleK[8][4]
    //   [8288..12384)    sK[8][512] (as float)
    //   [12384..16480)   kv_f[8][512] (roped, staged)
    //   [16480..16608)   summed logits [16][8]
    float* __restrict__ dbg
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    static_assert(HEAD_DIM % NPAL == 0 && SUB % 32 == 0, "bands must be 32-dim MMA chunks");
    static_assert(ROPE_DIM % 2 == 0, "interleaved RoPE needs even rope dim");

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

    if (n_slices == 0) {
        emit_partials();
        return;
    }

    uint8_t* write_slice_ptr =
        get_slice_mut<HEAD_DIM>(slices_ptr, (int)slot.write_slice, 1);
    const int ws_offset = (int)slice_offset(write_slice_ptr);
    const int ws_len = (int)slice_len(write_slice_ptr);
    const int q_pos = (int)slice_rope(write_slice_ptr) + ws_len;

    // ─── Fused single-latent scatter (warp 0): write this token's pre-RoPE
    // latent into the writer chunk's FP8 band arenas. K≡V → K bands only. ───
    {
        const int within = ws_offset + ws_len;
        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, 0);
            const T* src = kv_new + (int64_t)slot_idx * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) {
                int d = lane * DPT + j;
                int band = d / SUB;
                int in_band = d % SUB;
                uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM>(head_ptr, band);
                if (band_ptr != 0) {
                    __nv_fp8_e4m3* dst = (__nv_fp8_e4m3*)(uintptr_t)band_ptr;
                    dst[(int64_t)within * SUB + in_band] =
                        __nv_fp8_e4m3(to_f32<T>(src[d]));
                }
            }
        }
        __syncthreads();
    }

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
                float f = rope_freqs[(d - NOPE_DIM) >> 1];
                rope_pair(qr[j], qr[j + 1], q_pos, f);
            }
        }
        // Band max over the 4 threads covering (head, band): threads t..t^3.
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(qr[j]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        float s = mx / 127.f;
        if (s == 0.f) s = 1.f;
        const int band = (tid % 16) / 4;
        if ((tid & 3) == 0) scaleQ[head_local][band] = s;
        float inv = 1.f / s;
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
    auto load_tile = [&](int tile_idx) {
        bool valid = false;
        int key_pos = 0;
        float regs[DPT];
        if (tile_idx < n_win_tiles) {
            // Window source: FP8 band arenas.
            int sl_idx, within_base;
            tile_to_slice(tile_idx, sl_idx, within_base);
            int within = within_base + warp;
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, sl_idx, 1);
            int off = (int)slice_offset(sl);
            if (sl_idx < (int)n_slices && within < off + slice_eff_len(sl_idx)) {
                key_pos = (int)slice_rope(sl) + (within - off);
                // Sliding-window + causal bound (exact regardless of chunk
                // granularity).
                if (key_pos <= q_pos && key_pos > q_pos - window_size) {
                    valid = true;
                    const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, 0);
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        int d = lane * DPT + j;
                        int band = d / SUB;
                        uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM>(head_ptr, band);
                        float outer = kvhead_k_scale<HEAD_DIM>(head_ptr, band);
                        const uint8_t* src = (const uint8_t*)(uintptr_t)band_ptr;
                        float v = band_ptr
                            ? fp8_to_f32(src[(int64_t)within * SUB + (d % SUB)]) / outer
                            : 0.f;
                        regs[j] = v;
                    }
                }
            }
        } else {
            // Compressed source: f32 gallery rows via the selection list.
            int e = (tile_idx - n_win_tiles) * KEYS_TILE + warp;
            if (e < (int)n_sel) {
                uint32_t gid = comp_idx[(int64_t)slot_idx * max_sel + e];
                if (gid != 0xFFFFFFFFu) {
                    valid = true;
                    key_pos = (int)comp_pos[gid];
                    const float* src = comp + (int64_t)gid * HEAD_DIM;
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) regs[j] = src[lane * DPT + j];
                }
            }
        }
        if (!valid) {
            #pragma unroll
            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
        }
        // RoPE at the key's own position (pairs are lane-local: 16-dim
        // segments are even-aligned).
        if (valid) {
            #pragma unroll
            for (int j = 0; j < DPT; j += 2) {
                int d = lane * DPT + j;
                if (d >= NOPE_DIM) {
                    float f = rope_freqs[(d - NOPE_DIM) >> 1];
                    rope_pair(regs[j], regs[j + 1], key_pos, f);
                }
            }
        }
        // Stage FP latent (the PV read; K≡V) + per-band int8 (the QK read).
        #pragma unroll
        for (int j = 0; j < DPT; ++j)
            kv_f[warp][lane * DPT + j] = from_f32<T>(regs[j]);
        {
            float mx = 0.f;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) mx = fmaxf(mx, fabsf(regs[j]));
            // Band = lane/8 (lanes [8b, 8b+8) cover band b's 128 dims).
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
            float s = mx / 127.f;
            if (s == 0.f) s = 1.f;
            if ((lane & 7) == 0) scaleK[warp][lane / 8] = s;
            float inv = 1.f / s;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) {
                float v = fminf(fmaxf(regs[j] * inv, -127.f), 127.f);
                sK[warp][lane * DPT + j] = (int8_t)__float2int_rn(v);
            }
        }
        if (lane == 0) key_valid[warp] = valid ? 1 : 0;
    };

    const bool dump = dbg != nullptr && slot_idx == 0 && blockIdx.y == 0 && split_idx == 0;

    // ─── Main tile loop ───────────────────────────────────────────────────
    for (int tile = tile_lo; tile < tile_hi; ++tile) {
        load_tile(tile);
        __syncthreads();

        if (dump && tile == tile_lo && tid == 0) {
            for (int h = 0; h < HEADS_TILE; ++h)
                for (int p = 0; p < NPAL; ++p) dbg[h * NPAL + p] = scaleQ[h][p];
            for (int h = 0; h < HEADS_TILE; ++h)
                for (int d = 0; d < HEAD_DIM; ++d)
                    dbg[64 + h * HEAD_DIM + d] = (float)sQ[h][d];
            for (int t = 0; t < KEYS_TILE; ++t)
                for (int p = 0; p < NPAL; ++p) dbg[8256 + t * NPAL + p] = scaleK[t][p];
            for (int t = 0; t < KEYS_TILE; ++t)
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dbg[8288 + t * HEAD_DIM + d] = (float)sK[t][d];
                    dbg[12384 + t * HEAD_DIM + d] = to_f32<T>(kv_f[t][d]);
                }
        }

        // QK: warps 0-3, one 128-dim band each; 4 k-steps of m16n8k32
        // accumulated int32 (uniform scale within the band), then scaled to
        // float partial scores.
        if (warp < NPAL) {
            const int p = warp;
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
            for (int h = 0; h < HEADS_TILE; ++h)
                for (int t = 0; t < KEYS_TILE; ++t)
                    dbg[16480 + h * KEYS_TILE + t] =
                        scores_p[0][h][t] + scores_p[1][h][t] + scores_p[2][h][t]
                        + scores_p[3][h][t];
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
                float lg = scores_p[0][head_local][t] + scores_p[1][head_local][t]
                         + scores_p[2][head_local][t] + scores_p[3][head_local][t];
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
// Prefill kernel — the decode body re-shaped for many queries over a settled
// arena. The host pre-writes every fresh latent into the slot (contiguous
// write + usage commit) BEFORE the launch, so a query reads purely from the
// arena: no fused scatter, no writer +1 — causality is the per-query position
// clamp. One block = (query token, 16-head tile, split); positions come from
// a device array and the compressed selection is per query.
//
// This is the correctness-first prefill (identical numerics to running the
// decode kernel once per token, amortizing nothing): the flash-style Q-tile
// staging that amortizes KV traffic across queries is a measured optimization
// for the batched core, not a semantic change.
// =============================================================================
template <typename T, int HEAD_DIM, int ROPE_DIM>
__global__ void __launch_bounds__(WARPS * 32, 4)
deepseek_prefill_kernel(
    const T* __restrict__ q,               // [total_q, H, HEAD_DIM] pre-RoPE
    const uint8_t* __restrict__ headers,   // SlotHeader[1] — the slot
    const uint32_t* __restrict__ q_pos,    // [total_q]
    // Fresh-token key source: latents computed THIS layer (not yet in the
    // arena — the host writes them after the layer completes). Row j sits at
    // position fresh_base + j; query qi attends fresh rows causally like any
    // other window key. NULL/0 on the settled-slot path.
    const T* __restrict__ kv_fresh,        // [fresh_rows, HEAD_DIM] pre-RoPE
    const float* __restrict__ comp,        // [G_total, HEAD_DIM] pre-RoPE
    const uint32_t* __restrict__ comp_pos, // [G_total]
    const uint32_t* __restrict__ comp_idx, // [total_q, max_sel] ascending
    const uint32_t* __restrict__ comp_cnt, // [total_q]
    const float* __restrict__ rope_freqs,  // [ROPE_DIM/2]
    float* __restrict__ partial_acc,       // [total_q*H, splits, HEAD_DIM]
    float* __restrict__ partial_ml,        // [total_q*H, splits, 2]
    int total_q,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int fresh_rows,
    int fresh_base
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
    (void)fresh_base; // fresh positions come from q_pos (rows are queries)

    const int qi = (int)blockIdx.x;
    const int head_base = (int)blockIdx.y * HEADS_TILE;
    const int split_idx = (int)blockIdx.z;
    const int num_splits = (int)gridDim.z;
    const int tid = (int)threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (qi >= total_q) return;

    constexpr int DPT = HEAD_DIM / 32;
    float m_i[2] = {-1e38f, -1e38f};
    float l_i[2] = {0.f, 0.f};
    float out_reg[2][DPT];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int j = 0; j < DPT; ++j) out_reg[h][j] = 0.f;

    auto emit_partials = [&]() {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            int head = head_base + 2 * warp + h;
            if (head >= n_q_head) continue;
            int64_t base = ((int64_t)qi * n_q_head + head) * num_splits + split_idx;
            float* acc = partial_acc + base * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) acc[lane * DPT + j] = out_reg[h][j];
            if (lane == 0) {
                partial_ml[base * 2] = m_i[h];
                partial_ml[base * 2 + 1] = l_i[h];
            }
        }
    };

    const SlotHeader& slot = get_slot_header(headers, 0);
    const uint32_t n_slices = slot.n_slices;
    const uint64_t slices_ptr = slot.slices_ptr;
    const uint32_t n_sel = comp_cnt ? comp_cnt[qi] : 0;
    const int my_pos = (int)q_pos[qi];

    if (n_slices == 0 && n_sel == 0 && fresh_rows == 0) {
        emit_partials();
        return;
    }

    __shared__ alignas(128) int8_t sQ[HEADS_TILE][HEAD_DIM];
    __shared__ alignas(16) float scaleQ[HEADS_TILE][NPAL];
    __shared__ alignas(128) T kv_f[KEYS_TILE][HEAD_DIM];
    __shared__ alignas(128) int8_t sK[KEYS_TILE][HEAD_DIM];
    __shared__ alignas(16) float scaleK[KEYS_TILE][NPAL];
    __shared__ alignas(16) float scores_p[NPAL][HEADS_TILE][KEYS_TILE];
    __shared__ int key_valid[KEYS_TILE];

    // Q stage (rope at my_pos; same thread geometry as decode).
    {
        const int head_local = tid / 16;
        const int dseg = (tid % 16) * 32;
        const int head = head_base + head_local;
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
            if (d >= NOPE_DIM) {
                float f = rope_freqs[(d - NOPE_DIM) >> 1];
                rope_pair(qr[j], qr[j + 1], my_pos, f);
            }
        }
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) mx = fmaxf(mx, fabsf(qr[j]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        float s = mx / 127.f;
        if (s == 0.f) s = 1.f;
        const int band = (tid % 16) / 4;
        if ((tid & 3) == 0) scaleQ[head_local][band] = s;
        float inv = 1.f / s;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float v = fminf(fmaxf(qr[j] * inv, -127.f), 127.f);
            sQ[head_local][dseg + j] = (int8_t)__float2int_rn(v);
        }
    }
    __syncthreads();

    // Window tiling: committed lengths only (the host wrote + committed every
    // fresh token before launch — no writer +1).
    auto slice_len_of = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, 1);
        return (int)slice_len(sl);
    };
    auto slice_tiles = [&](int s) -> int {
        return (slice_len_of(s) + KEYS_TILE - 1) / KEYS_TILE;
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
    const int n_fresh_tiles = (fresh_rows + KEYS_TILE - 1) / KEYS_TILE;
    const int n_comp_tiles = ((int)n_sel + KEYS_TILE - 1) / KEYS_TILE;
    const int n_tiles = n_win_tiles + n_fresh_tiles + n_comp_tiles;

    const int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
    int tile_lo = split_idx * tiles_per_split;
    int tile_hi = tile_lo + tiles_per_split;
    if (tile_lo > n_tiles) tile_lo = n_tiles;
    if (tile_hi > n_tiles) tile_hi = n_tiles;

    auto load_tile = [&](int tile_idx) {
        bool valid = false;
        int key_pos = 0;
        float regs[DPT];
        if (tile_idx < n_win_tiles) {
            int sl_idx, within_base;
            tile_to_slice(tile_idx, sl_idx, within_base);
            int within = within_base + warp;
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, sl_idx, 1);
            int off = (int)slice_offset(sl);
            if (sl_idx < (int)n_slices && within < off + slice_len_of(sl_idx)) {
                key_pos = (int)slice_rope(sl) + (within - off);
                if (key_pos <= my_pos && key_pos > my_pos - window_size) {
                    valid = true;
                    const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, 0);
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        int d = lane * DPT + j;
                        int band = d / SUB;
                        uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM>(head_ptr, band);
                        float outer = kvhead_k_scale<HEAD_DIM>(head_ptr, band);
                        const uint8_t* src = (const uint8_t*)(uintptr_t)band_ptr;
                        regs[j] = band_ptr
                            ? fp8_to_f32(src[(int64_t)within * SUB + (d % SUB)]) / outer
                            : 0.f;
                    }
                }
            }
        } else if (tile_idx < n_win_tiles + n_fresh_tiles) {
            // Fresh-token source: this layer's just-computed latents, read
            // straight from the input buffer through an FP8 ROUND-TRIP so the
            // key bits match what every later wave will read from the arena.
            // Fresh rows ARE the query rows, so key j's position is q_pos[j] —
            // contiguous for prefill, arbitrary for glue islands.
            int fj = (tile_idx - n_win_tiles) * KEYS_TILE + warp;
            if (fj < fresh_rows) {
                key_pos = (int)q_pos[fj];
                if (key_pos <= my_pos && key_pos > my_pos - window_size) {
                    valid = true;
                    const T* src = kv_fresh + (int64_t)fj * HEAD_DIM;
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) {
                        float v = to_f32<T>(src[lane * DPT + j]);
                        __nv_fp8_e4m3 enc = __nv_fp8_e4m3(v);
                        regs[j] = fp8_to_f32(*reinterpret_cast<uint8_t*>(&enc));
                    }
                }
            }
        } else {
            int e = (tile_idx - n_win_tiles - n_fresh_tiles) * KEYS_TILE + warp;
            if (e < (int)n_sel) {
                uint32_t gid = comp_idx[(int64_t)qi * max_sel + e];
                if (gid != 0xFFFFFFFFu) {
                    valid = true;
                    key_pos = (int)comp_pos[gid];
                    const float* src = comp + (int64_t)gid * HEAD_DIM;
                    #pragma unroll
                    for (int j = 0; j < DPT; ++j) regs[j] = src[lane * DPT + j];
                }
            }
        }
        if (!valid) {
            #pragma unroll
            for (int j = 0; j < DPT; ++j) regs[j] = 0.f;
        }
        if (valid) {
            #pragma unroll
            for (int j = 0; j < DPT; j += 2) {
                int d = lane * DPT + j;
                if (d >= NOPE_DIM) {
                    float f = rope_freqs[(d - NOPE_DIM) >> 1];
                    rope_pair(regs[j], regs[j + 1], key_pos, f);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < DPT; ++j)
            kv_f[warp][lane * DPT + j] = from_f32<T>(regs[j]);
        {
            float mx = 0.f;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) mx = fmaxf(mx, fabsf(regs[j]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
            float s = mx / 127.f;
            if (s == 0.f) s = 1.f;
            if ((lane & 7) == 0) scaleK[warp][lane / 8] = s;
            float inv = 1.f / s;
            #pragma unroll
            for (int j = 0; j < DPT; ++j) {
                float v = fminf(fmaxf(regs[j] * inv, -127.f), 127.f);
                sK[warp][lane * DPT + j] = (int8_t)__float2int_rn(v);
            }
        }
        if (lane == 0) key_valid[warp] = valid ? 1 : 0;
    };

    for (int tile = tile_lo; tile < tile_hi; ++tile) {
        load_tile(tile);
        __syncthreads();

        if (warp < NPAL) {
            const int p = warp;
            int32_t c[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int ks = 0; ks < SUB / 32; ++ks) {
                uint32_t a_frag[4];
                uint32_t b_frag[2];
                fused_attn::load_a_frag_m16k32(a_frag, &sQ[0][p * SUB + ks * 32], HEAD_DIM, lane);
                fused_attn::load_b_frag_n8k32(b_frag, &sK[0][p * SUB + ks * 32], HEAD_DIM, lane);
                fused_attn::mma_int8_m16n8k32(c, a_frag, b_frag, c);
            }
            const int r0 = lane >> 2;
            const int c0 = (lane & 3) * 2;
            scores_p[p][r0][c0]     = (float)c[0] * scaleQ[r0][p] * scaleK[c0][p];
            scores_p[p][r0][c0 + 1] = (float)c[1] * scaleQ[r0][p] * scaleK[c0 + 1][p];
            scores_p[p][r0 + 8][c0]     = (float)c[2] * scaleQ[r0 + 8][p] * scaleK[c0][p];
            scores_p[p][r0 + 8][c0 + 1] = (float)c[3] * scaleQ[r0 + 8][p] * scaleK[c0 + 1][p];
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int head_local = 2 * warp + h;
            if (head_base + head_local >= n_q_head) continue;
            float sc[KEYS_TILE];
            float tile_max = -1e38f;
            #pragma unroll
            for (int t = 0; t < KEYS_TILE; ++t) {
                float lg = scores_p[0][head_local][t] + scores_p[1][head_local][t]
                         + scores_p[2][head_local][t] + scores_p[3][head_local][t];
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
// Glue latent scatter: write `rows` latents into their RESERVED gap chunks
// (block index + in-block offset per row, from the reprojection's PendingGlue
// descriptors). Launched stream-ordered BEFORE the attention pass, so glue
// keys read from the arena like any window key — no double-source, no
// intra-launch race. One warp per row.
// =============================================================================
template <typename T, int HEAD_DIM>
__global__ void deepseek_glue_scatter_kernel(
    const T* __restrict__ kv,             // [rows, HEAD_DIM] pre-RoPE latents
    const uint8_t* __restrict__ headers,  // SlotHeader[1] — the slot
    const uint32_t* __restrict__ slices,  // [rows] gap block index
    const uint32_t* __restrict__ in_blk,  // [rows] in-block offset
    int rows
) {
    constexpr int SUB = HEAD_DIM / NPAL;
    constexpr int DPT = HEAD_DIM / 32;
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = (int)threadIdx.x % 32;
    if (row >= rows) return;
    const SlotHeader& slot = get_slot_header(headers, 0);
    const uint8_t* slp =
        get_slice<HEAD_DIM>(slot.slices_ptr, (int)slices[row], 1);
    const uint8_t* head_ptr = get_head<HEAD_DIM>(slp, 0);
    const T* src = kv + (int64_t)row * HEAD_DIM;
    int within = (int)in_blk[row];
    #pragma unroll
    for (int j = 0; j < DPT; ++j) {
        int d = lane * DPT + j;
        int band = d / SUB;
        uint64_t band_ptr = kvhead_k_ptr<HEAD_DIM>(head_ptr, band);
        if (band_ptr != 0) {
            __nv_fp8_e4m3* dst = (__nv_fp8_e4m3*)(uintptr_t)band_ptr;
            dst[(int64_t)within * SUB + (d % SUB)] = __nv_fp8_e4m3(to_f32<T>(src[d]));
        }
    }
}

// Prefill combine: as the decode combine, but the query position comes from
// the per-query array (no writer-slice derivation).
template <typename O, int HEAD_DIM, int ROPE_DIM>
__global__ void deepseek_prefill_combine_kernel(
    O* __restrict__ out,                    // [total_q, H, HEAD_DIM]
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_ml,
    const uint32_t* __restrict__ q_pos,     // [total_q]
    const float* __restrict__ sinks,        // [H]
    const float* __restrict__ rope_freqs,
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
        float f = rope_freqs[(d - NOPE_DIM) >> 1];
        float r, c, s;
        int k;
        rope_angle((int)q_pos[qi], f, r, k);
        ds_sincos(r, k, s, c);
        val = (d & 1) == 0
            ? __fadd_rn(__fmul_rn(val, c), __fmul_rn(partner, s))
            : __fsub_rn(__fmul_rn(val, c), __fmul_rn(partner, s));
    }
    out[(int64_t)row * HEAD_DIM + d] = from_f32<O>(val);
}

// =============================================================================
// Combine: merge split partials, fold the per-head sink, normalize, de-rotate
// the output's rope dims at the query position, write the final output.
// One block per (slot, head) row; HEAD_DIM threads.
// =============================================================================
template <typename O, int HEAD_DIM, int ROPE_DIM>
__global__ void deepseek_combine_kernel(
    O* __restrict__ out,                    // [slots, H, HEAD_DIM]
    const float* __restrict__ partial_acc,  // [rows, splits, HEAD_DIM]
    const float* __restrict__ partial_ml,   // [rows, splits, 2]
    const uint8_t* __restrict__ headers,
    const float* __restrict__ sinks,        // [H]
    const float* __restrict__ rope_freqs,   // [ROPE_DIM/2]
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
        const SlotHeader& slot = get_slot_header(headers, slot_idx);
        int q_pos = 0;
        if (slot.n_slices > 0) {
            const uint8_t* ws =
                get_slice<HEAD_DIM>(slot.slices_ptr, (int)slot.write_slice, 1);
            q_pos = (int)slice_rope(ws) + (int)slice_len(ws);
        }
        float partner = __shfl_xor_sync(0xffffffff, val, 1);
        float f = rope_freqs[(d - NOPE_DIM) >> 1];
        float r, c, s;
        int k;
        rope_angle(q_pos, f, r, k);
        ds_sincos(r, k, s, c);
        // inverse: even' = x0·c + x1·s ; odd' = x1·c − x0·s (explicit-rounded)
        val = (d & 1) == 0
            ? __fadd_rn(__fmul_rn(val, c), __fmul_rn(partner, s))
            : __fsub_rn(__fmul_rn(val, c), __fmul_rn(partner, s));
    }

    out[(int64_t)row * HEAD_DIM + d] = from_f32<O>(val);
}

// Grow-on-demand partial pool (fork-local; the stock pool lives in the
// fused_attn namespace with the stock kernels compiled into its TU).
inline void deepseek_partial_pool(
    int64_t rows, int splits, int head_dim, float** acc_out, float** ml_out,
    cudaStream_t stream
) {
    static float* g_acc = nullptr;
    static float* g_ml = nullptr;
    static int64_t g_cap_acc = 0;
    static int64_t g_cap_ml = 0;
    int64_t need_acc = rows * splits * head_dim;
    int64_t need_ml = rows * splits * 2;
    if (need_acc > g_cap_acc) {
        if (g_acc) { cudaStreamSynchronize(stream); cudaFree(g_acc); }
        if (cudaMalloc(&g_acc, (size_t)need_acc * sizeof(float)) != cudaSuccess) {
            g_acc = nullptr; g_cap_acc = 0; *acc_out = nullptr; *ml_out = nullptr; return;
        }
        g_cap_acc = need_acc;
    }
    if (need_ml > g_cap_ml) {
        if (g_ml) { cudaStreamSynchronize(stream); cudaFree(g_ml); }
        if (cudaMalloc(&g_ml, (size_t)need_ml * sizeof(float)) != cudaSuccess) {
            g_ml = nullptr; g_cap_ml = 0; *acc_out = nullptr; *ml_out = nullptr; return;
        }
        g_cap_ml = need_ml;
    }
    *acc_out = g_acc;
    *ml_out = g_ml;
}

inline int deepseek_sm_count() {
    static int sm = 0;
    if (sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        if (sm <= 0) sm = 1;
    }
    return sm;
}

template <typename T, int HEAD_DIM, int ROPE_DIM>
void launch_deepseek_decode(
    const T* q,
    const uint8_t* headers,
    T* out,
    const T* kv_new,
    const float* comp,
    const uint32_t* comp_pos,
    const uint32_t* comp_idx,
    const uint32_t* comp_cnt,
    const float* sinks,
    const float* rope_freqs,
    int num_slots,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int num_splits_override,  // > 0 pins the split factor (test determinism)
    bool commit_write_len,    // advance the header write-len on-device (live buffer)
    cudaStream_t stream,
    float* dbg = nullptr      // nullable stage-dump (see kernel doc)
) {
    if (num_slots <= 0 || n_q_head <= 0) return;
    const int head_tiles = (n_q_head + HEADS_TILE - 1) / HEADS_TILE;

    int num_splits;
    if (num_splits_override > 0) {
        num_splits = num_splits_override;
    } else {
        int base_blocks = num_slots * head_tiles;
        int target_blocks = deepseek_sm_count() * 4 * 2;
        num_splits = (target_blocks + base_blocks - 1) / base_blocks;
    }
    if (num_splits < 1) num_splits = 1;
    if (num_splits > 32) num_splits = 32;

    float* pa = nullptr;
    float* pm = nullptr;
    deepseek_partial_pool((int64_t)num_slots * n_q_head, num_splits, HEAD_DIM,
                          &pa, &pm, stream);
    if (pa == nullptr) return;  // allocation failure: nothing launched

    dim3 grid(num_slots, head_tiles, num_splits);
    dim3 block(WARPS * 32);
    deepseek_decode_kernel<T, HEAD_DIM, ROPE_DIM><<<grid, block, 0, stream>>>(
        q, headers, kv_new, comp, comp_pos, comp_idx, comp_cnt, rope_freqs,
        pa, pm, num_slots, n_q_head, softmax_scale, window_size, max_sel, dbg);

    const int num_rows = num_slots * n_q_head;
    deepseek_combine_kernel<T, HEAD_DIM, ROPE_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
        out, pa, pm, headers, sinks, rope_freqs, num_rows, n_q_head, num_splits);

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

template <typename T, int HEAD_DIM, int ROPE_DIM>
void launch_deepseek_prefill(
    const T* q,
    const uint8_t* headers,
    T* out,
    const uint32_t* q_pos,
    const T* kv_fresh,
    const float* comp,
    const uint32_t* comp_pos,
    const uint32_t* comp_idx,
    const uint32_t* comp_cnt,
    const float* sinks,
    const float* rope_freqs,
    int total_q,
    int n_q_head,
    float softmax_scale,
    int window_size,
    int max_sel,
    int fresh_rows,
    int fresh_base,
    int num_splits_override,
    cudaStream_t stream
) {
    if (total_q <= 0 || n_q_head <= 0) return;
    const int head_tiles = (n_q_head + HEADS_TILE - 1) / HEADS_TILE;
    int num_splits = (num_splits_override > 0) ? num_splits_override : 1;
    if (num_splits > 32) num_splits = 32;

    float* pa = nullptr;
    float* pm = nullptr;
    deepseek_partial_pool((int64_t)total_q * n_q_head, num_splits, HEAD_DIM, &pa, &pm, stream);
    if (pa == nullptr) return;

    dim3 grid(total_q, head_tiles, num_splits);
    dim3 block(WARPS * 32);
    deepseek_prefill_kernel<T, HEAD_DIM, ROPE_DIM><<<grid, block, 0, stream>>>(
        q, headers, q_pos, kv_fresh, comp, comp_pos, comp_idx, comp_cnt, rope_freqs,
        pa, pm, total_q, n_q_head, softmax_scale, window_size, max_sel,
        fresh_rows, fresh_base);

    const int num_rows = total_q * n_q_head;
    deepseek_prefill_combine_kernel<T, HEAD_DIM, ROPE_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
        out, pa, pm, q_pos, sinks, rope_freqs, num_rows, n_q_head, num_splits);
}

}  // namespace deepseek_attn
