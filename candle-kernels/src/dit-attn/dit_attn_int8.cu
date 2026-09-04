/*
 * ============================================================================
 * DENSE INT8 ATTENTION FOR DIFFUSION TRANSFORMERS
 * ============================================================================
 *
 * `softmax(q·kᵀ/√d)·v` over a flat sequence, **no mask**, computed with INT8
 * m16n8k32 tensor-core MMA for both products — the same "compressed domain is
 * the compute domain" idea as `paged-prefill/paged_prefill_int8_kernel.cuh`,
 * for the workload that kernel cannot serve.
 *
 * # Why a second int8 attention kernel
 *
 * The paged one is causal (`horizon = prefix_len + token + 1`, computed per
 * tile rather than passed in) and reads K/V from the palette-quantized paged
 * arena through chunk headers. A diffusion transformer is neither: its sequence
 * is image patches followed by caption tokens with no order to be causal about,
 * and its K/V are projections computed fresh for one block of one denoise step.
 *
 * Everything hard in that kernel is therefore absent here — no palettes, no
 * rank tables, no chunk straddling, no GQA packing, no causal horizon — and
 * what is left is reused: the MMA wrappers, the fragment loaders, and the
 * online-softmax shape.
 *
 * # Quantization grid
 *
 *   Q:  int8 per ROW (over the whole head_dim)
 *   K:  int8 per TOKEN (over the whole head_dim), post-RoPE
 *   P:  int8 per row, fixed scale 1/127
 *   V:  int8 per DIM (over the sequence), mean-centred
 *
 *   QK epilogue: score = i32(whole 128-deep dot) · qs[row] · ks[tok]
 *   PV epilogue: o_f32 = o_f32·α + i32 · (1/127) · vs[dim]
 *   store:       out   = o_f32 / l[row] + vmean[dim]
 *
 * **Per-row scales, not the paged kernel's per-32-window.** Measured on the
 * model's own tensors, the two are within 2% of each other (rel_l2 0.0192 vs
 * 0.0194 against an f32 oracle), and per-row is much the better shape: one
 * int32 accumulator runs the entire 128-deep dot across all four MMAs and takes
 * a single fixup, where a per-window grid needs a fresh accumulator and a
 * convert-and-scale every 32 elements. The paged kernel needs the fine grid
 * because it consumes whatever the arena's palette produced; this one quantizes
 * its own operands and can choose.
 *
 * **V is centred and K is not**, which is the reverse of the usual advice and
 * follows from the architecture: this model applies `q_norm` and `k_norm` but
 * has no `v_norm`, so Q and K arrive per-head RMSNorm'd with no channel
 * outliers left to fix, while V carries whatever per-channel bias its
 * projection gives it. Measured: centring V takes a block's error from 0.0642
 * to 0.0192; centring K takes it to 0.0640. Both centrings are exact rather
 * than approximate — subtracting a constant from every key shifts a row's
 * scores by one value that softmax ignores, and `P(V−μ) = PV−μ` because the
 * softmax rows sum to one — but only V's is worth the pass. Both identities
 * need every row to attend over every key, which is true here precisely
 * because there is no mask.
 *
 * # Layout
 *
 *   q8    [B, H, S, D] int8      qs    [B, H, S] f32
 *   k8    [B, H, S, D] int8      ks    [B, H, S] f32
 *   v8    [B, H, D, S] int8      vs    [B, H, D] f32   vmean [B, H, D] f32
 *   out   [B, H, S, D] bf16
 *
 * V is stored TRANSPOSED because the `.row.col` MMA reads its B operand as
 * `[n][k]`, and for the P·V product `n` is the output dim and `k` is the token.
 * That is also why V's scale must be per dim: a per-token scale would vary
 * along the MMA's K and could not be folded out of an int32 accumulator, while
 * a per-dim one varies along N and folds into the epilogue for free.
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <limits>

#include "../mma/mma_wrappers.cuh"

namespace {

/// The running max's identity element.
///
/// Spelled out rather than `-INFINITY`, which under `--use_fast_math` expands to
/// `1e+300` — a double that overflows to the right float, with a warning at every
/// use to say it did. This is the value that was meant.
constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

/// Threads per quantizer block — eight warps, one row each.
constexpr int QUANT_THREADS = 256;

// One warp owns 16 query rows — the MMA's M.
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 32;

/// Smem row stride for a `HEAD_DIM`-wide int8 tile.
///
/// Padded off a 128-byte multiple so the eight rows a B-fragment load touches
/// land in distinct banks: at stride 144 row `r` starts at bank `(4r) % 32`, so
/// rows 0..7 open banks 0,4,…,28 and each row's four lanes take the four banks
/// above it — a conflict-free 32-lane read.
template <int HEAD_DIM>
struct Pad {
    static constexpr int K_STRIDE = HEAD_DIM + 16;
};

/// `softmax(q·kᵀ)·v`, one block per (query tile, head, batch).
///
/// `BLOCK_M` query rows are split one 16-row tile per warp; `BLOCK_N` keys are
/// staged per iteration. The O accumulator lives in registers across the whole
/// key loop, which is what makes this a flash kernel rather than a tiled one:
/// the `[BLOCK_M, S]` score matrix is never written anywhere.
template <int HEAD_DIM, int BLOCK_M, int BLOCK_N, int WARPS>
// **`__launch_bounds__` names the block size and NOT a minimum block count.**
//
// Asking ptxas for 4 blocks/SM caps it at 128 registers, which it reaches by
// spilling: 180 bytes of spill stores and loads and a 112-byte stack frame.
// Measured, that trade is badly negative — 117.5 TOP/s at 168 registers and 3
// blocks/SM against 85.2 at 128 registers and 4 (−28%), and the same 29% at
// 512×512. Occupancy is the wrong target here: the O accumulator is 64 f32
// registers per warp because a 16×128 f32 tile *is* that size, and the kernel
// already reaches 86% of the card's int8 ceiling with 25% occupancy on
// instruction-level parallelism. More resident warps would only help if it were
// latency-bound, and at 86% of peak arithmetic it is not.
__global__ __launch_bounds__(WARPS * 32) void dit_attn_int8_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ qs,
    const int8_t* __restrict__ k8,
    const float* __restrict__ ks,
    const int8_t* __restrict__ v8,
    const float* __restrict__ vs,
    const float* __restrict__ vmean,
    __nv_bfloat16* __restrict__ out,
    int seq,
    int v_stride)
{
    constexpr int K_STRIDE = Pad<HEAD_DIM>::K_STRIDE;
    constexpr int V_STRIDE = BLOCK_N + 16;
    constexpr int N_TILES  = BLOCK_N / MMA_N;   // score column tiles
    constexpr int D_TILES  = HEAD_DIM / MMA_N;  // output column tiles
    constexpr int QK_CHUNKS = HEAD_DIM / MMA_K; // 128 / 32 = 4
    constexpr int PV_CHUNKS = BLOCK_N / MMA_K;

    // **16-byte aligned, explicitly.** A `__shared__ int8_t[]` is aligned to its
    // element type — one byte — so the 16-byte staging stores and the 4-byte
    // fragment loads below both fault on it. The strides are multiples of 16;
    // only the base needed saying.
    __shared__ __align__(16) int8_t s_k[BLOCK_N * K_STRIDE];
    __shared__ __align__(16) int8_t s_v[HEAD_DIM * V_STRIDE];
    __shared__ __align__(16) int8_t s_p[WARPS * MMA_M * V_STRIDE];
    __shared__ float  s_ks[BLOCK_N];
    __shared__ float  s_vs[HEAD_DIM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int bh   = blockIdx.y + blockIdx.z * gridDim.y;   // flattened (batch, head)
    const int m0   = blockIdx.x * BLOCK_M + warp * MMA_M;   // this warp's first row

    const int64_t seq_base = (int64_t)bh * seq;
    const int8_t* q_bh = q8 + seq_base * HEAD_DIM;
    const int8_t* k_bh = k8 + seq_base * HEAD_DIM;
    const int8_t* v_bh = v8 + (int64_t)bh * HEAD_DIM * v_stride;

    // The C-fragment's own decomposition: lane `l` holds rows `l>>2` and
    // `(l>>2)+8` of the 16, at columns `(l&3)*2` and `+1`.
    const int g  = lane >> 2;
    const int n0 = (lane & 3) * 2;
    const int r0 = m0 + g;
    const int r1 = m0 + g + 8;

    // Per-row Q scales, once. Rows past the end read zero and are never stored.
    const float qs0 = (r0 < seq) ? qs[seq_base + r0] : 0.f;
    const float qs1 = (r1 < seq) ? qs[seq_base + r1] : 0.f;

    // **Q's fragments are loaded once and kept.** Q is invariant across the key
    // loop, so it costs 4 registers per k-chunk and saves re-reading a tile that
    // never changes — the same reason the paged kernel drains its Q into
    // registers before the loop.
    uint32_t q_frag[QK_CHUNKS][4];
    {
        // A-fragment lanes: row `lane>>2` and `+8`, four bytes at `(lane&3)*4`.
        const int ar = lane >> 2;
        const int ac = (lane & 3) * 4;
        #pragma unroll
        for (int c = 0; c < QK_CHUNKS; ++c) {
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + ar + h * 8;
                uint32_t lo = 0, hi = 0;
                if (row < seq) {
                    const int8_t* p = q_bh + (int64_t)row * HEAD_DIM + c * MMA_K + ac;
                    lo = *reinterpret_cast<const uint32_t*>(p);
                    hi = *reinterpret_cast<const uint32_t*>(p + 16);
                }
                q_frag[c][h]     = lo;
                q_frag[c][h + 2] = hi;
            }
        }
    }

    // V's per-dim scale and mean are loop-invariant; the scale carries P's fixed
    // 1/127 folded in so the PV epilogue is one multiply.
    for (int d = tid; d < HEAD_DIM; d += WARPS * 32) {
        s_vs[d] = vs[(int64_t)bh * HEAD_DIM + d] * (1.f / 127.f);
    }

    float o_acc[D_TILES][4];
    #pragma unroll
    for (int n = 0; n < D_TILES; ++n)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[n][i] = 0.f;

    float m_run[2] = { NEG_INF, NEG_INF };
    float l_run[2] = { 0.f, 0.f };

    for (int kv0 = 0; kv0 < seq; kv0 += BLOCK_N) {
        __syncthreads();
        // Stage K `[BLOCK_N][HEAD_DIM]` and V `[HEAD_DIM][BLOCK_N]`. Both are
        // 16-byte vector copies: HEAD_DIM and BLOCK_N are multiples of 16, and
        // int8 rows are naturally aligned.
        constexpr int VEC = 16;
        constexpr int K_VECS = BLOCK_N * (HEAD_DIM / VEC);
        for (int i = tid; i < K_VECS; i += WARPS * 32) {
            const int t = i / (HEAD_DIM / VEC);
            const int c = (i % (HEAD_DIM / VEC)) * VEC;
            const int tok = kv0 + t;
            int4 val = make_int4(0, 0, 0, 0);
            if (tok < seq) {
                val = *reinterpret_cast<const int4*>(k_bh + (int64_t)tok * HEAD_DIM + c);
            }
            *reinterpret_cast<int4*>(&s_k[t * K_STRIDE + c]) = val;
        }
        constexpr int V_VECS = HEAD_DIM * (BLOCK_N / VEC);
        for (int i = tid; i < V_VECS; i += WARPS * 32) {
            const int d = i / (BLOCK_N / VEC);
            const int t = (i % (BLOCK_N / VEC)) * VEC;
            // `v_stride` is `seq` rounded up to 16, so a row's 16-byte reads are
            // aligned for any sequence length and the vector path never has a
            // scalar fallback. The quantizer zeroes the pad, so a tail read
            // brings in zeros — which is what a zeroed P would multiply anyway.
            const int4 val =
                *reinterpret_cast<const int4*>(v_bh + (int64_t)d * v_stride + kv0 + t);
            *reinterpret_cast<int4*>(&s_v[d * V_STRIDE + t]) = val;
        }
        for (int t = tid; t < BLOCK_N; t += WARPS * 32) {
            const int tok = kv0 + t;
            s_ks[t] = (tok < seq) ? ks[seq_base + tok] : 0.f;
        }
        __syncthreads();

        // ── Q·Kᵀ ─────────────────────────────────────────────────────────
        //
        // One int32 accumulator per column tile runs the WHOLE 128-deep dot —
        // four MMAs, no intermediate convert — because Q's scale is per row and
        // K's is per token, so neither varies along the reduction. That is the
        // per-row grid paying for itself.
        float sc[N_TILES][4];
        #pragma unroll
        for (int s = 0; s < N_TILES; ++s) {
            int32_t acc[4] = { 0, 0, 0, 0 };
            #pragma unroll
            for (int c = 0; c < QK_CHUNKS; ++c) {
                uint32_t b[2];
                // `ldmatrix` rather than four strided 4-byte reads: one
                // instruction per fragment, and the padded strides above are
                // exactly what it needs to stay conflict-free.
                fused_attn::load_b_frag_n8k32_ldmatrix(
                    b, &s_k[(s * MMA_N) * K_STRIDE + c * MMA_K], K_STRIDE, lane);
                fused_attn::mma_int8_m16n8k32(acc, q_frag[c], b, acc);
            }
            const float k0 = s_ks[s * MMA_N + n0];
            const float k1 = s_ks[s * MMA_N + n0 + 1];
            const int c0 = kv0 + s * MMA_N + n0;
            const int c1 = c0 + 1;
            // A key past the end contributes nothing: -inf leaves the running
            // max alone and exponentiates to zero.
            sc[s][0] = (r0 < seq && c0 < seq) ? (float)acc[0] * qs0 * k0 : NEG_INF;
            sc[s][1] = (r0 < seq && c1 < seq) ? (float)acc[1] * qs0 * k1 : NEG_INF;
            sc[s][2] = (r1 < seq && c0 < seq) ? (float)acc[2] * qs1 * k0 : NEG_INF;
            sc[s][3] = (r1 < seq && c1 < seq) ? (float)acc[3] * qs1 * k1 : NEG_INF;
        }

        // ── online softmax ───────────────────────────────────────────────
        //
        // A row lives in four lanes (those sharing `lane>>2`), two columns each,
        // so the row reduction is a butterfly over lane bits 0 and 1.
        float m_new[2], alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            float m_tile = NEG_INF;
            #pragma unroll
            for (int s = 0; s < N_TILES; ++s) {
                m_tile = fmaxf(m_tile, sc[s][row * 2]);
                m_tile = fmaxf(m_tile, sc[s][row * 2 + 1]);
            }
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 2));
            m_new[row] = fmaxf(m_run[row], m_tile);
            alpha[row] = (m_run[row] == NEG_INF) ? 0.f : __expf(m_run[row] - m_new[row]);
        }

        // P = exp(score − rowmax) ∈ (0, 1], so the 1/127 scale needs no scan.
        // The normaliser accumulates the EXACT float; only the MMA's operand is
        // rounded.
        float l_add[2] = { 0.f, 0.f };
        int8_t* p_row = &s_p[warp * MMA_M * V_STRIDE];
        #pragma unroll
        for (int s = 0; s < N_TILES; ++s) {
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const float a = (m_new[row] == NEG_INF)
                                    ? 0.f
                                    : __expf(sc[s][row * 2] - m_new[row]);
                const float b = (m_new[row] == NEG_INF)
                                    ? 0.f
                                    : __expf(sc[s][row * 2 + 1] - m_new[row]);
                l_add[row] += a + b;
                // `n0` is even, so the pair stores as one aligned 16-bit write.
                const uint16_t pk =
                    (uint16_t)(uint8_t)(int8_t)__float2int_rn(a * 127.f) |
                    ((uint16_t)(uint8_t)(int8_t)__float2int_rn(b * 127.f) << 8);
                *(uint16_t*)&p_row[(g + row * 8) * V_STRIDE + s * MMA_N + n0] = pk;
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

        // ── P·V ──────────────────────────────────────────────────────────
        //
        // **P's fragments are hoisted out of the dim loop.** They depend on the
        // k-chunk alone, not on which output dims are being accumulated, so
        // loading them inside would read the same bytes back out of shared
        // memory once per dim tile — sixteen times over, per key tile, per warp.
        uint32_t p_frag[PV_CHUNKS][4];
        #pragma unroll
        for (int c = 0; c < PV_CHUNKS; ++c) {
            fused_attn::load_a_frag_m16k32_ldmatrix(
                p_frag[c], &p_row[c * MMA_K], V_STRIDE, lane);
        }
        #pragma unroll
        for (int n = 0; n < D_TILES; ++n) {
            int32_t acc[4] = { 0, 0, 0, 0 };
            #pragma unroll
            for (int c = 0; c < PV_CHUNKS; ++c) {
                uint32_t b[2];
                fused_attn::load_b_frag_n8k32_ldmatrix(
                    b, &s_v[(n * MMA_N) * V_STRIDE + c * MMA_K], V_STRIDE, lane);
                fused_attn::mma_int8_m16n8k32(acc, p_frag[c], b, acc);
            }
            const float d0 = s_vs[n * MMA_N + n0];
            const float d1 = s_vs[n * MMA_N + n0 + 1];
            o_acc[n][0] = o_acc[n][0] * alpha[0] + (float)acc[0] * d0;
            o_acc[n][1] = o_acc[n][1] * alpha[0] + (float)acc[1] * d1;
            o_acc[n][2] = o_acc[n][2] * alpha[1] + (float)acc[2] * d0;
            o_acc[n][3] = o_acc[n][3] * alpha[1] + (float)acc[3] * d1;
        }
    }

    // ── epilogue ─────────────────────────────────────────────────────────
    //
    // Normalise by the exact running sum, then add V's mean back — the term
    // `P(V−μ) = PV−μ` left behind, exact because the softmax row sums to one.
    const float inv0 = (l_run[0] > 0.f) ? 1.f / l_run[0] : 0.f;
    const float inv1 = (l_run[1] > 0.f) ? 1.f / l_run[1] : 0.f;
    const float* mu = vmean + (int64_t)bh * HEAD_DIM;
    __nv_bfloat16* o_bh = out + seq_base * HEAD_DIM;
    #pragma unroll
    for (int n = 0; n < D_TILES; ++n) {
        const int d0 = n * MMA_N + n0;
        const int d1 = d0 + 1;
        if (r0 < seq) {
            o_bh[(int64_t)r0 * HEAD_DIM + d0] = __float2bfloat16(o_acc[n][0] * inv0 + mu[d0]);
            o_bh[(int64_t)r0 * HEAD_DIM + d1] = __float2bfloat16(o_acc[n][1] * inv0 + mu[d1]);
        }
        if (r1 < seq) {
            o_bh[(int64_t)r1 * HEAD_DIM + d0] = __float2bfloat16(o_acc[n][2] * inv1 + mu[d0]);
            o_bh[(int64_t)r1 * HEAD_DIM + d1] = __float2bfloat16(o_acc[n][3] * inv1 + mu[d1]);
        }
    }
}

/// Symmetric int8 per row, one warp per row — the one quantizer all three
/// operands use.
///
/// **V is this kernel too, because a transposed V's per-dim scale IS a per-row
/// scale.** Transposing V to `[dim, token]` is required anyway (the PV MMA reads
/// its B operand as `[n][k]`), and doing it *first* turns "reduce down a column
/// with a `head_dim`-strided read" into "reduce along a row" — which is the
/// difference between every lane touching its own 32-byte sector and a warp
/// reading one contiguous run. That is the whole reason this kernel has a
/// `CENTER` parameter rather than V having a kernel of its own.
///
/// `CENTER` subtracts the row mean and reports it, for the operand whose
/// projection carries a per-channel bias nothing normed away.
///
/// `dst_stride` may exceed `cols`; the tail is zeroed, which is what lets the
/// attention kernel read V in whole 16-byte vectors at any sequence length.
template <bool CENTER>
__global__ __launch_bounds__(QUANT_THREADS) void dit_quant_rows_kernel(
    const __nv_bfloat16* __restrict__ src,
    int8_t* __restrict__ dst,
    float* __restrict__ scale,
    float* __restrict__ mean_out,
    int rows,
    int cols,
    int dst_stride)
{
    // **A warp per row, but eight warps per block.** One warp per *block* is the
    // obvious shape and it is the wrong one: Q and K are 123,840 rows here, so
    // that launches 123,840 blocks of 32 threads, caps the SM at its 16-block
    // limit — 16 warps of 48 — and pays block setup for every row.
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (QUANT_THREADS / 32) + warp;
    if (row >= rows) return;
    const int64_t base = (int64_t)row * cols;
    int8_t* out = dst + (int64_t)row * dst_stride;

    // **Four bf16 at a time where the row allows it.** A `head_dim` of 128 is
    // exactly 32 lanes × 4, so a row's statistics come from one 8-byte load per
    // lane instead of four 2-byte ones. The row stride is `cols`, so the vector
    // path needs `cols % 4 == 0` for the base of every row to stay 8-byte
    // aligned; a ragged sequence takes the scalar loop instead. The test is
    // uniform across the block, so the branch itself costs nothing.
    const bool vec = (cols & 3) == 0;
    const int groups = cols >> 2;

    // **One statistics pass, not two.** Centring needs the mean before it can
    // take the centred amax, which reads as two passes — but the largest
    // deviation from a mean is attained at an extreme, so
    // `max|v−μ| = max(vmax−μ, μ−vmin)` exactly, and sum/min/max all come from
    // the same read.
    float sum = 0.f, vmin = -NEG_INF, vmax = NEG_INF, amax = 0.f;
    auto see = [&](float x) {
        if (CENTER) {
            sum += x;
            vmin = fminf(vmin, x);
            vmax = fmaxf(vmax, x);
        } else {
            amax = fmaxf(amax, fabsf(x));
        }
    };
    if (vec) {
        for (int g = lane; g < groups; g += 32) {
            const __nv_bfloat162* p =
                reinterpret_cast<const __nv_bfloat162*>(src + base + g * 4);
            const __nv_bfloat162 a = p[0], b = p[1];
            see(__bfloat162float(a.x));
            see(__bfloat162float(a.y));
            see(__bfloat162float(b.x));
            see(__bfloat162float(b.y));
        }
    } else {
        for (int c = lane; c < cols; c += 32) {
            see(__bfloat162float(src[base + c]));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        if (CENTER) {
            sum += __shfl_xor_sync(0xffffffffu, sum, off);
            vmin = fminf(vmin, __shfl_xor_sync(0xffffffffu, vmin, off));
            vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, off));
        } else {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
    }
    float mu = 0.f;
    if (CENTER) {
        mu = sum / (float)cols;
        amax = fmaxf(vmax - mu, mu - vmin);
    }

    const float inv = (amax != 0.f) ? 127.f / amax : 0.f;
    if (lane == 0) {
        scale[row] = amax / 127.f;
        if (CENTER) mean_out[row] = mu;
    }
    auto q1 = [&](float x) -> int8_t {
        const float q = fminf(127.f, fmaxf(-127.f, rintf((x - mu) * inv)));
        return (int8_t)q;
    };
    if (vec) {
        // Four quantized bytes land as one 32-bit store, matching the 8-byte
        // load that produced them.
        for (int g = lane; g < groups; g += 32) {
            const __nv_bfloat162* p =
                reinterpret_cast<const __nv_bfloat162*>(src + base + g * 4);
            const __nv_bfloat162 a = p[0], b = p[1];
            const char4 v4 = make_char4(q1(__bfloat162float(a.x)), q1(__bfloat162float(a.y)),
                                        q1(__bfloat162float(b.x)), q1(__bfloat162float(b.y)));
            *reinterpret_cast<char4*>(out + g * 4) = v4;
        }
    } else {
        for (int c = lane; c < cols; c += 32) {
            out[c] = q1(__bfloat162float(src[base + c]));
        }
    }
    // The pad the attention kernel's last 16-byte read runs into.
    for (int c = cols + lane; c < dst_stride; c += 32) {
        out[c] = (int8_t)0;
    }
}

} // namespace

// The one geometry this model runs. A second head_dim adds an instantiation
// here rather than a template parameter at the call site: the tile constants
// below are chosen against it, and a silently-wrong tiling is wrong attention
// rather than a fault.
constexpr int DIT_HEAD_DIM = 128;
constexpr int DIT_BLOCK_M = 64;
constexpr int DIT_BLOCK_N = 64;
constexpr int DIT_WARPS = 4;

extern "C" void run_dit_attn_int8_bf16(
    const void* q8, const void* qs,
    const void* k8, const void* ks,
    const void* v8, const void* vs, const void* vmean,
    void* out,
    int batch, int heads, int seq, int head_dim, int v_stride,
    void* stream)
{
    if (head_dim != DIT_HEAD_DIM) return;
    dim3 grid((seq + DIT_BLOCK_M - 1) / DIT_BLOCK_M, heads, batch);
    dit_attn_int8_kernel<DIT_HEAD_DIM, DIT_BLOCK_M, DIT_BLOCK_N, DIT_WARPS>
        <<<grid, DIT_WARPS * 32, 0, (cudaStream_t)stream>>>(
            (const int8_t*)q8, (const float*)qs,
            (const int8_t*)k8, (const float*)ks,
            (const int8_t*)v8, (const float*)vs, (const float*)vmean,
            (__nv_bfloat16*)out, seq, v_stride);
}

extern "C" void run_dit_quant_rows_bf16(
    const void* src, void* dst, void* scale, void* mean,
    int rows, int cols, int dst_stride, int center, void* stream)
{
    const int warps = QUANT_THREADS / 32;
    const int blocks = (rows + warps - 1) / warps;
    if (center) {
        dit_quant_rows_kernel<true><<<blocks, QUANT_THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)src, (int8_t*)dst, (float*)scale, (float*)mean,
            rows, cols, dst_stride);
    } else {
        dit_quant_rows_kernel<false><<<blocks, QUANT_THREADS, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16*)src, (int8_t*)dst, (float*)scale, (float*)mean,
            rows, cols, dst_stride);
    }
}
