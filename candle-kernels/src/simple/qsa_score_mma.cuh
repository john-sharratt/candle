// =============================================================================
// QSA index scoring on the INT8 tensor cores
// =============================================================================
// The paged scorer's wide arm (`qsa_score_paged.cu`), built the way the decode
// attention kernel computes QK^T (`../paged-decode/int8_decode_kernel.cuh`):
//
//   * the QUERY tile is quantized to int8 once a block, one scale per
//     32-channel block of each `(row, head)`;
//   * a WARP holds one key collectively — four channels a lane, so a Q8_0
//     block is eight lanes — dequantizes its rotary half, rotates it at the
//     key's own position, and REQUANTIZES it to int8 against the rotated
//     block's own maximum, found by three shuffle rounds. The pass-through
//     half is the stored codes as they stand;
//   * the dot products are `mma.sync` INT8: sixteen `(row, head)` rows against
//     eight keys over one 32-channel block an instruction (two 16-channel
//     instructions on sm_86), and each integer result is scaled by
//     `s_q · s_k` into the float the heads are folded from.
//
// A lane then holds four codes to a register where the float arms hold one
// channel: 71 registers against the float `(4, 4, 2)` arm's 128, and three
// blocks an SM against its two — what its shared memory allows.
//
// Geometry: `D = 128` (four Q8_0 blocks, 32 words a key) and a rotary width of
// 0, 64 or 128 channels — the released indexer's. `H` heads divide an MMA's
// sixteen rows into `16 / H` query rows, and a block holds `QSA_MMA_TILES`
// such tiles. Anything else takes the generic arm.
//
// Numerics: the query is rounded to int8 and the rotated half of a key is
// rounded a second time, so a cell differs from the float score by at most
// `Σ_h Σ_b (½·s_k·Σ|q| + ½·s_q·Σ|k| + 8·s_q·s_k)` over the rotary blocks and
// `Σ_h Σ_b ½·s_q·Σ|k|` over the pass-through ones — the bound
// `tests/qsa_score_rot_harness.rs` derives and holds it to.

#pragma once

#include "../mma/mma_wrappers.cuh"

#define QSA_MMA_WARPS 8
#define QSA_MMA_THREADS (QSA_MMA_WARPS * 32)
/// Keys a warp stages a pass: the MMA's N.
#define QSA_MMA_KEYS 8
/// `(row, head)` rows one MMA scores: its M.
#define QSA_MMA_ROWS 16
/// MMA row tiles a block holds. **Staging a key is the cost, not scoring it**:
/// a key is dequantized, rotated and requantized by a whole warp, and an MMA
/// tile then scores eight of them against sixteen rows in a handful of
/// instructions. Held to one tile — four query rows of four heads — every
/// four rows restaged every key and the kernel ran 1.1–1.8× the float
/// scorer's time; a block's keys are staged once for all of its tiles.
#define QSA_MMA_TILES 8
/// `(row, head)` rows a block holds.
#define QSA_MMA_TILE_ROWS (QSA_MMA_ROWS * QSA_MMA_TILES)
#define QSA_MMA_D 128
/// Words of four codes in a key.
#define QSA_MMA_WORDS (QSA_MMA_D / 4)
#define QSA_MMA_NB (QSA_MMA_D / QSA_Q8_BLOCK)
/// The widest rotary width, in pairs.
#define QSA_MMA_MAX_PAIRS 64
/// Blocks per SM the kernel is compiled to fit — what its shared memory
/// allows: 80 registers.
#define QSA_MMA_MIN_BLOCKS 3

/// Four values as one word of int8 codes at `inv = 1 / scale`.
__device__ __forceinline__ int qsa_mma_pack(float4 v, float inv)
{
    const int a = __float2int_rn(fminf(fmaxf(v.x * inv, -127.f), 127.f)) & 0xff;
    const int b = __float2int_rn(fminf(fmaxf(v.y * inv, -127.f), 127.f)) & 0xff;
    const int c = __float2int_rn(fminf(fmaxf(v.z * inv, -127.f), 127.f)) & 0xff;
    const int d = __float2int_rn(fminf(fmaxf(v.w * inv, -127.f), 127.f)) & 0xff;
    return a | (b << 8) | (c << 16) | (d << 24);
}

/// One page's descriptor, as a warp's lanes all hold it.
struct QsaMmaPage {
    const signed char* codes;
    const unsigned short* scales;
    int cstride;
    int rstride;
    long long delta;
    int first;
};

__device__ __forceinline__ QsaMmaPage qsa_mma_page(
    const long long* __restrict__ pages, const unsigned int* __restrict__ page_first, int p)
{
    const long long* pd = pages + (long long)p * QSA_PAGED_PAGE_WORDS;
    QsaMmaPage out;
    out.codes = reinterpret_cast<const signed char*>((uintptr_t)__ldg(pd));
    out.scales = reinterpret_cast<const unsigned short*>((uintptr_t)__ldg(pd + 1));
    out.cstride = (int)__ldg(pd + 2);
    out.rstride = (int)__ldg(pd + 3);
    out.delta = __ldg(pd + 4);
    out.first = (int)__ldg(page_first + p);
    return out;
}

template <int H>
__global__ __launch_bounds__(QSA_MMA_THREADS, QSA_MMA_MIN_BLOCKS) void qsa_score_paged_mma_kernel(
    const float* __restrict__ q,                     // [rows*H, D], rotated
    const long long* __restrict__ pages,             // [P * PAGE_WORDS]
    const unsigned int* __restrict__ page_first,     // [P+1] exclusive prefix
    const unsigned int* __restrict__ cnt,            // [rows] valid prefix
    const float2* __restrict__ tab,                  // factored RoPE table
    const float2* __restrict__ steps,                // [pairs][32] step table
    float* __restrict__ out,                         // [rows, n_cand], `out_s` apart
    int rows, int n_cand, int P, int pairs, int ratio,
    long long out_s, long long row_base)
{
    // The query tile as int8 words and its block scales; each warp's staged
    // keys and theirs; and the step table's first `KEYS` columns — a warp's
    // keys are that many rows apart at most.
    //
    // **The operand rows are padded by a word.** An MMA fragment load has its
    // eight lane groups read the same words of eight consecutive rows; at a
    // row pitch of exactly 32 words those are one bank, an 8-way conflict on
    // every load. One word of padding puts each row a bank along.
    __shared__ int s_q8[QSA_MMA_TILE_ROWS][QSA_MMA_WORDS + 1];
    __shared__ float s_qs[QSA_MMA_TILE_ROWS][QSA_MMA_NB];
    __shared__ int s_k8[QSA_MMA_WARPS][QSA_MMA_KEYS][QSA_MMA_WORDS + 1];
    __shared__ float s_ks[QSA_MMA_WARPS][QSA_MMA_KEYS][QSA_MMA_NB];
    //
    // **The steps are two planes, key-major.** A lane reads the entries of its
    // four pairs, and the eight lanes of a block hold pairs four apart. As
    // `float2`s a pair's eight keys apart, those were 64 words apart — one
    // bank, a 6-way conflict on 81% of the kernel's shared loads (`ncu`). As a
    // sine and a cosine plane with a key's pairs side by side, the eight lanes
    // read words four apart and take eight banks.
    __shared__ float s_step_sin[QSA_MMA_KEYS][QSA_MMA_MAX_PAIRS];
    __shared__ float s_step_cos[QSA_MMA_KEYS][QSA_MMA_MAX_PAIRS];

    constexpr int TR = QSA_MMA_TILE_ROWS / H;
    const int r0 = blockIdx.y * TR;
    const int nr = min(TR, rows - r0);
    if (nr <= 0) return;
    // MMA row tiles the block's query rows fill; the last may be part zeros.
    const int n_tiles = (nr * H + QSA_MMA_ROWS - 1) / QSA_MMA_ROWS;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // ---- the query tile, quantized: a warp is one block of one row ---------
    // Element `e` is channel `e & 127` of MMA row `e >> 7`, and its lane is its
    // channel within the block, so the block's maximum is five shuffle rounds
    // and four lanes' codes pack into one word. A row past the tile is zeros.
    for (int e = threadIdx.x; e < n_tiles * QSA_MMA_ROWS * QSA_MMA_D; e += QSA_MMA_THREADS) {
        const int m = e >> 7;
        const int c = e & (QSA_MMA_D - 1);
        const bool live = m < nr * H;
        const float x = live ? q[(long long)(r0 * H + m) * QSA_MMA_D + c] : 0.f;
        float amax = fabsf(x);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(QSA_FULL, amax, off));
        }
        float s = amax * (1.f / 127.f);
        if (s == 0.f) s = 1.f;
        const int code = __float2int_rn(fminf(fmaxf(x / s, -127.f), 127.f)) & 0xff;
        const int w = code | (__shfl_down_sync(QSA_FULL, code, 1) << 8)
            | (__shfl_down_sync(QSA_FULL, code, 2) << 16)
            | (__shfl_down_sync(QSA_FULL, code, 3) << 24);
        if ((lane & 3) == 0) s_q8[m][c >> 2] = w;
        if (lane == 0) s_qs[m][c / QSA_Q8_BLOCK] = live ? s : 0.f;
    }
    for (int i = threadIdx.x; i < pairs * QSA_MMA_KEYS; i += QSA_MMA_THREADS) {
        const int n = i / pairs;
        const int pr = i - n * pairs;
        const float2 st = __ldg(steps + pr * 32 + n);
        s_step_sin[n][pr] = st.x;
        s_step_cos[n][pr] = st.y;
    }
    __syncthreads();

    // A lane's place in a key: its word of four channels, its Q8_0 block, and
    // — in the rotary width — its half and its first pair. The halves are
    // `RL` lanes apart, so a lane's partner is `lane ^ RL`.
    const int RL = pairs >> 2;
    const bool rotary = lane < 2 * RL;
    // The rotary lanes: a whole number of eight-lane blocks, both halves.
    const unsigned int rot_mask = RL >= 16 ? QSA_FULL : ((1u << (2 * RL)) - 1u);
    const bool low = lane < RL;
    const int pair0 = RL > 0 ? (lane & (RL - 1)) * 4 : 0;
    const int blk = lane >> 3;
    const int row = lane >> 2;
    const int tq = lane & 3;
    const int n0 = tq * 2;

    for (int G = (blockIdx.x * QSA_MMA_WARPS + warp) * QSA_MMA_KEYS; G < n_cand;
         G += gridDim.x * QSA_MMA_WARPS * QSA_MMA_KEYS) {
        // ---- stage the warp's keys ----------------------------------------
        // Eight consecutive candidates are on one page unless a page ends
        // among them. On one page a key's rotation is the first key's composed
        // with a step, so the table is read once a pass; across pages each key
        // looks its own up. A candidate past the end restages the last key,
        // and its result is dropped.
        const int g_last = min(G + QSA_MMA_KEYS - 1, n_cand - 1);
        // One search a pass: the page holding the first key holds them all when
        // the last is below the next page's first row.
        int p = qsa_page_of(page_first, P, (unsigned int)G);
        const bool one_page = (unsigned int)g_last < __ldg(page_first + p + 1);
        QsaMmaPage pg = qsa_mma_page(pages, page_first, p);
        float2 w0[4];
        if (one_page && rotary) {
            const int pos0 = (int)(pg.delta + (long long)G * ratio);
            #pragma unroll
            for (int e = 0; e < 4; ++e) w0[e] = rope_f_lookup(tab, pairs, pos0, pair0 + e);
        }
        #pragma unroll
        for (int n = 0; n < QSA_MMA_KEYS; ++n) {
            const int g = min(G + n, n_cand - 1);
            if (!one_page) {
                p = qsa_page_of(page_first, P, (unsigned int)g);
                pg = qsa_mma_page(pages, page_first, p);
            }
            const int local = g - pg.first;
            const signed char* kb = pg.codes + (long long)local * pg.rstride;
            const int codes = __ldg(reinterpret_cast<const int*>(kb + row * pg.cstride) + tq);
            const float sk =
                qsa_q8_scale(__ldg(pg.scales + (long long)local * QSA_MMA_NB + blk));
            // A pass-through lane's word is its stored codes and its block's
            // scale the stored one: it stages them and does nothing else. Only
            // the rotary lanes dequantize, and their shuffles name only them.
            int word = codes;
            float scale = sk;
            if (rotary) {
                float4 v = qsa_q8_word(codes, sk);
                float4 o;
                o.x = __shfl_xor_sync(rot_mask, v.x, RL);
                o.y = __shfl_xor_sync(rot_mask, v.y, RL);
                o.z = __shfl_xor_sync(rot_mask, v.z, RL);
                o.w = __shfl_xor_sync(rot_mask, v.w, RL);
                float2 sc[4];
                if (one_page) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        float2 st;
                        st.x = s_step_sin[n][pair0 + e];
                        st.y = s_step_cos[n][pair0 + e];
                        sc[e] = qsa_compose(w0[e], st);
                    }
                } else {
                    const int pos = (int)(pg.delta + (long long)g * ratio);
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) sc[e] = rope_f_lookup(tab, pairs, pos, pair0 + e);
                }
                // The low half keeps the pair's first value, the high half its
                // second.
                if (low) {
                    rope_f_rotate(v.x, o.x, sc[0]);
                    rope_f_rotate(v.y, o.y, sc[1]);
                    rope_f_rotate(v.z, o.z, sc[2]);
                    rope_f_rotate(v.w, o.w, sc[3]);
                } else {
                    rope_f_rotate(o.x, v.x, sc[0]);
                    rope_f_rotate(o.y, v.y, sc[1]);
                    rope_f_rotate(o.z, v.z, sc[2]);
                    rope_f_rotate(o.w, v.w, sc[3]);
                }
                // The rotated block is requantized against its own maximum —
                // eight lanes, three rounds.
                float amax =
                    fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
                amax = fmaxf(amax, __shfl_xor_sync(rot_mask, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(rot_mask, amax, 2));
                amax = fmaxf(amax, __shfl_xor_sync(rot_mask, amax, 4));
                scale = amax * (1.f / 127.f);
                if (scale == 0.f) scale = 1.f;
                word = qsa_mma_pack(v, 1.f / scale);
            }
            s_k8[warp][n][lane] = word;
            if ((lane & 7) == 0) s_ks[warp][n][blk] = scale;
        }
        __syncwarp();

        // ---- the staged keys against every row tile of the block ----------
        // Sixteen rows against eight keys an MMA, a Q8_0 block at a time. The
        // result fragment gives this lane `(row, n0)`, `(row, n0 + 1)`,
        // `(row + 8, n0)` and `(row + 8, n0 + 1)` of the tile.
        for (int tile = 0; tile < n_tiles; ++tile) {
            const int m0 = tile * QSA_MMA_ROWS + row;     // this lane's first row
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int b = 0; b < QSA_MMA_NB; ++b) {
                int32_t c[4] = {0, 0, 0, 0};
#if __CUDA_ARCH__ >= 890
                const uint32_t a[4] = {
                    (uint32_t)s_q8[m0][b * 8 + tq], (uint32_t)s_q8[m0 + 8][b * 8 + tq],
                    (uint32_t)s_q8[m0][b * 8 + 4 + tq], (uint32_t)s_q8[m0 + 8][b * 8 + 4 + tq]};
                const uint32_t k[2] = {
                    (uint32_t)s_k8[warp][row][b * 8 + tq],
                    (uint32_t)s_k8[warp][row][b * 8 + 4 + tq]};
                fused_attn::mma_int8_m16n8k32(c, a, k, c);
#else
                #pragma unroll
                for (int kc = 0; kc < 2; ++kc) {
                    const int at = b * 8 + kc * 4 + tq;
                    const uint32_t a[2] = {(uint32_t)s_q8[m0][at], (uint32_t)s_q8[m0 + 8][at]};
                    const uint32_t k[1] = {(uint32_t)s_k8[warp][row][at]};
                    fused_attn::mma_int8_m16n8k16(c, a, k, c);
                }
#endif
                const float qa = s_qs[m0][b];
                const float qb = s_qs[m0 + 8][b];
                const float ka = s_ks[warp][n0][b];
                const float kb2 = s_ks[warp][n0 + 1][b];
                acc[0] += (float)c[0] * (qa * ka);
                acc[1] += (float)c[1] * (qa * kb2);
                acc[2] += (float)c[2] * (qb * ka);
                acc[3] += (float)c[3] * (qb * kb2);
            }

            // ReLU, then fold a query row's heads. MMA row `m` is head
            // `m % H` of query row `m / H` — a tile is a whole number of query
            // rows — and rows are four lanes apart, so a row's heads are
            // `4, 8, …` lanes away.
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[i] = fmaxf(acc[i], 0.f);
            #pragma unroll
            for (int hb = 1; hb < H; hb <<= 1) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) acc[i] += __shfl_xor_sync(QSA_FULL, acc[i], hb * 4);
            }
            if ((row & (H - 1)) == 0) {
                const int qa = m0 / H;
                const int qb = (m0 + 8) / H;
                #pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int g = G + n0 + t;
                    if (g >= n_cand) continue;
                    if (qa < nr) {
                        out[(row_base + r0 + qa) * out_s + g] =
                            (unsigned int)g < __ldg(cnt + r0 + qa) ? acc[t] : QSA_PAGED_MASK;
                    }
                    if (qb < nr) {
                        out[(row_base + r0 + qb) * out_s + g] =
                            (unsigned int)g < __ldg(cnt + r0 + qb) ? acc[2 + t] : QSA_PAGED_MASK;
                    }
                }
            }
        }
        // The next pass restages this warp's keys.
        __syncwarp();
    }
}

/// The fewest query rows, and the fewest `rows × candidates` cells (candidates
/// rounded up to a power of two, as the grid is), the tensor-core arm takes.
/// Measured on sm_86 against the float arms: it wins from about four million
/// cells — 2.0× at 512 rows over 128K tokens — and loses below, and at eight
/// rows it loses at every depth.
#define QSA_MMA_MIN_ROWS 16
#define QSA_MMA_MIN_CELLS (1LL << 22)

/// Blocks the grid holds per SM. A block quantizes its query tile before it
/// scores anything, so it has to score many passes of keys to be worth
/// starting: the grid is the blocks the device runs at once and a few more to
/// even the tail, and each strides the candidate axis. One block per pass of
/// keys redid the tile for every 64 keys and spent more on it than on the
/// scores.
#define QSA_MMA_BLOCKS_PER_SM 8

/// Launch the tensor-core arm at `HV` heads.
#define QSA_MMA_LAUNCH(HV)                                                          \
    case HV: {                                                                      \
        const int span = QSA_MMA_WARPS * QSA_MMA_KEYS;                              \
        const int tr = QSA_MMA_TILE_ROWS / (HV);                                    \
        const int by = (rows + tr - 1) / tr;                                        \
        const int cap = (QSA_MMA_BLOCKS_PER_SM * qsa_paged_sm_count() + by - 1) / by; \
        int bx = (n_grid + span - 1) / span;                                        \
        if (bx > cap) bx = cap;                                                     \
        if (bx < 1) bx = 1;                                                         \
        dim3 grid(bx, by);                                                          \
        qsa_score_paged_mma_kernel<HV><<<grid, QSA_MMA_THREADS, 0, stream>>>(       \
            q, pages, page_first, cnt, tab2, steps2, out, rows, n_cand, P, pairs,   \
            ratio, out_s, row_base);                                                \
        return;                                                                     \
    }
