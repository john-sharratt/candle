#pragma once
// Gated DeltaNet — fused prefill scan (the chunked parallel form).
//
// The arithmetic is `delta_chunked` in
// candle-transformers/src/models/delta_net/mix.rs (itself parity-locked to
// the sequential rule). Per chunk of C tokens, with entering state
// `S [d_v, d_k]` per V head and within-chunk log-decay cumsum G:
//
//   g_t     = a_h · softplus(α_t + dt_bias_h),  β_t = σ(βlin_t)
//   D[i][j] = exp(min(0, G[i] − G[j]))                    (decay, clamped)
//   A[i][j] = β_i (k_i · k_j) D[i][j]        for j < i    (strictly lower)
//   (I + A) [u | w] = [βv | βk ⊙ e^G]                     (one fwd-subst solve)
//   v_new   = u − w Sᵀ                                    (chunk-local writes)
//   o[i]    = e^{G[i]} (q_i Sᵀ) + Σ_{j ≤ i} (q_i·k_j) D[i][j] v_new[j]
//   S       ← e^{G[C−1]} S + v_newᵀ (k ⊙ e^{G[C−1] − G})  (in place, as stored)
//
// Why this is not the decode kernel in a loop: the decode step touches the
// full state once per token, which is correct at t == 1 and quadratic-in-state
// traffic at t > 1. The chunked form pays the state once per chunk; these
// kernels keep it in registers across the whole sequence.
//
// **The kernels read the mixer's own buffers through strides** — there is no
// GQA repeat, no per-span contiguous copy, no separate q/k/v tensors:
//   qk     : [T, 2·h_k, D]  l2-normed Q|K stack; V head h reads K head
//            h % h_k (ggml's tiled broadcast — §7.8 of the design doc), and
//            q is scaled by `q_scale` on load.
//   v      : a strided view into the post-SiLU conv output — base pointer at
//            the V column offset, token stride = conv_dim.
//   α, βlin: [T, h_v] raw projections; the gates are computed in-kernel from
//            dt_bias/a (softplus and sigmoid exactly as the reference: the
//            stable `max(x,0) + log1p(e^{−|x|})` form).
//   o      : written into the caller's whole-wave output at the span's rows,
//            so a multi-sequence wave needs no concatenation.
//
// Three kernels, C = 64 (the width at which the triangle lives in smem):
//   conv_prefill — token-parallel causal conv (unchanged layout contract).
//   intra        — per (V-head, chunk): gates, G scan, A build, one forward-
//                  substitution solve for both right-hand sides, and the
//                  inclusive dot grid kq.
//   state        — per (V-head, d_v-tile): the sequential chunk walk with the
//                  S tile register-resident in the stored orientation and the
//                  output fused.
// (The row-wise norm/SiLU-gate epilogue shared with the decode path lives in
// delta_net_common.cuh.)
//
// All state math is F32 (the state is an unbounded running sum — §7.16 of
// docs/qwen35_qwen38_models.md); no TF32. The exponent clamp on D is
// load-bearing: the discarded upper half of G[i] − G[j] grows positive with
// distance and overflows exp to +inf, and inf × 0 is NaN.
//
// Concrete (non-template) kernels: this header is compiled by the single
// translation unit delta_net_api_f32.cu; `static` keeps the definitions
// TU-local so a second includer cannot collide at link time.

#include "delta_net_common.cuh"

#define DNP_CHUNK 64
#define DNP_DIM DN_HEAD_DIM
#define DNP_THREADS 256
// Smem row strides padded +1 so column-of-row accesses spread across banks.
#define DNP_LD (DNP_DIM + 1)
#define DNP_ALD (DNP_CHUNK + 1)
// d_v rows of state owned by one state-pass block.
#define DNP_TV 32
// Tokens the state pass stages per half-chunk — half of DNP_CHUNK, so its
// stage buffer is half-size and two blocks fit an SM (see the kernel header).
#define DNP_TH 32

namespace delta_net {

// ============================================================================
// The prefill span table — the multi-sequence half of `DeltaNetLayerTable`.
//
// A wave carries one span per prefilling or verifying sequence, each with its
// own carried state and its own row range of the packed buffers. Launching one
// kernel per span is what the decode path already refuses to do (hot-path
// invariant 5): its states live in per-session allocations, so it takes their
// ADDRESSES on the device and runs the whole cohort in one launch. These spans
// are the same problem with two extra fields, so they take the same answer.
//
//   ptrs  [4, n] i64 — conv tail in, conv tail out, state in, state out
//   spans [2, n] u32 — first row in the packed wave buffer, row count
//
// The pointer rows are laid out exactly as the decode table's, so one builder
// serves both. `blockIdx.z` selects the span; every kernel below rebases its
// wave-buffer reads by `start` and bounds its work by `len`, which is what lets
// spans of different lengths share a launch.
// ============================================================================
struct DnSpan {
    const float* tail;
    float*       tail_out;
    const float* state;
    float*       state_out;
    int          start;
    int          len;
};

__device__ __forceinline__ DnSpan dn_span(
        const long long* __restrict__ ptrs,
        const unsigned int* __restrict__ spans,
        int n_spans,
        int z) {
    DnSpan s;
    s.tail      = reinterpret_cast<const float*>(ptrs[z]);
    s.tail_out  = reinterpret_cast<float*>(ptrs[n_spans + z]);
    s.state     = reinterpret_cast<const float*>(ptrs[2 * n_spans + z]);
    s.state_out = reinterpret_cast<float*>(ptrs[3 * n_spans + z]);
    s.start     = (int)spans[z];
    s.len       = (int)spans[n_spans + z];
    return s;
}

// ============================================================================
// Token-parallel causal conv with the SiLU + Q|K-norm epilogue: the output is
// the post-activation, post-norm buffer every downstream kernel reads q/k/v
// from through strides.
//   y[t][c] = epilogue( Σ_j kern[c][j] · in(t − (K−1) + j) )
// where in(p) reads x for p ≥ 0 and the entering tail for p < 0. The new tail
// stores the RAW inputs (pre-activation, as the conv window wants them) and
// goes to `tail_out` — never in place, because blocks computing outputs for
// t < K−1 are still reading the entering tail.
//
// One launch for every span in the wave: `blockIdx.z` picks the span, and
// `x`/`y` are the whole packed buffers rebased by its `start`.
// ============================================================================
static __global__ void delta_net_conv_prefill_f32_kernel(
        const float* __restrict__ x_wave,  // [T_wave, C]
        const float* __restrict__ kernel,  // [C, K]
        float*       __restrict__ y_wave,  // [T_wave, C]
        const long long*    __restrict__ ptrs,
        const unsigned int* __restrict__ spans,
        int n_spans,
        int channels,
        int kwidth,
        int qk_channels,
        float eps) {
    __shared__ float red[256];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const DnSpan sp = dn_span(ptrs, spans, n_spans, blockIdx.z);
    const int t = blockIdx.y;
    // The launch is a rectangle over the WIDEST span, so shorter spans leave
    // block rows with no token. `t` is `blockIdx.y`, so this is uniform across
    // the block and the epilogue's block-collective reduction below is never
    // entered by only part of a block.
    if (t >= sp.len) return;
    const float* __restrict__ x = x_wave + (size_t)sp.start * channels;
    float* __restrict__ y = y_wave + (size_t)sp.start * channels;
    const float* __restrict__ tail = sp.tail;
    float* __restrict__ tail_out = sp.tail_out;
    const int t_len = sp.len;
    const int tcols = kwidth - 1;
    const float* krow = kernel + (size_t)c * kwidth;

    float acc = 0.f;
    for (int j = 0; j < kwidth; ++j) {
        const int idx = t - tcols + j;
        const float val = (idx >= 0)
            ? x[(size_t)idx * channels + c]
            : tail[(size_t)c * tcols + (tcols + idx)];
        acc += krow[j] * val;
    }
    y[(size_t)t * channels + c] =
        dn_silu_norm_epilogue(acc, c, qk_channels, eps, (int)threadIdx.x, red);

    // One block row owns the tail write; its reads see only the old buffer.
    if (t == 0) {
        for (int j = 0; j < tcols; ++j) {
            const int idx = t_len - tcols + j;
            tail_out[(size_t)c * tcols + j] = (idx >= 0)
                ? x[(size_t)idx * channels + c]
                : tail[(size_t)c * tcols + (tcols + idx)];
        }
    }
}

// ============================================================================
// Intra-chunk kernel: one block per (chunk, V head).
//
// Dynamic smem partition (83.5 KB — the launcher opts in past the 48 KB
// default): sk [C][LD], sq [C][LD], A [C][ALD], G [C], β [C].
//
// The solve assigns one right-hand-side column per thread: d_v + d_k = 256
// columns = 256 threads exactly. X lives in registers with the 64-step
// substitution fully unrolled (static indices — a dynamic index would spill
// the array to local memory); A[i][j] reads at step i are the same row for
// every thread, i.e. smem broadcasts. Rows at and past c_len hold garbage in
// both A and X; they are never stored, and clean rows i < c_len only ever
// read x[j] for j < i, so the garbage stays confined to discarded lanes.
// ============================================================================
static __global__ void delta_net_prefill_intra_f32_kernel(
        const float* __restrict__ qk_wave,   // Q|K columns of the conv output
        const float* __restrict__ v_wave,    // base at the V column, same stride
        const float* __restrict__ alpha_wave,// [T_wave, h_v] raw
        const float* __restrict__ blin_wave, // [T_wave, h_v] raw
        const float* __restrict__ dt_bias,   // [h_v]
        const float* __restrict__ a_neg,     // [h_v]
        float*       __restrict__ u,         // [h_v, T_tran, D]
        float*       __restrict__ w,         // [h_v, T_tran, D]
        float*       __restrict__ kq,        // [h_v, T_tran, C]
        float*       __restrict__ g_cs,      // [h_v, T_tran]
        const unsigned int* __restrict__ spans,
        int n_spans,
        int t_tran,     // rows per head of the shared transients
        int n_v_heads,
        int n_k_heads,
        int tok_stride, // conv_dim: q, k and v are strided views of one buffer
        float q_scale) {
    extern __shared__ float smem[];
    float* sk = smem;                          // [C][LD]
    float* sq = sk + DNP_CHUNK * DNP_LD;       // [C][LD]
    float* sA = sq + DNP_CHUNK * DNP_LD;       // [C][ALD]
    float* sg = sA + DNP_CHUNK * DNP_ALD;      // [C] G cumsum
    float* sb = sg + DNP_CHUNK;                // [C] β

    const int z = blockIdx.z;
    const int span_start = (int)spans[z];
    const int t_len = (int)spans[n_spans + z];
    const int t0 = blockIdx.x * DNP_CHUNK;
    // The launch is a rectangle over the span with the most chunks; a shorter
    // span's surplus blocks have no tokens. Uniform across the block.
    if (t0 >= t_len) return;
    // Rebase the packed wave buffers so every index below is span-local — the
    // same arithmetic this kernel ran when it was launched once per span.
    const float* __restrict__ qk = qk_wave + (size_t)span_start * tok_stride;
    const float* __restrict__ v = v_wave + (size_t)span_start * tok_stride;
    const float* __restrict__ alpha = alpha_wave + (size_t)span_start * n_v_heads;
    const float* __restrict__ blin = blin_wave + (size_t)span_start * n_v_heads;
    // The transients are ONE wave-wide allocation per layer rather than one per
    // span, so a head's rows are `t_tran` apart and this span's sit at
    // `span_start` within them.
    const size_t tran = (size_t)span_start;

    const int h = blockIdx.y;
    const int kh = h % n_k_heads; // ggml's tiled GQA broadcast
    const int c_len = min(DNP_CHUNK, t_len - t0);
    const int tid = (int)threadIdx.x;
    const int qk_stride = tok_stride;

    // Gates for the chunk (computed here, not by the caller), then an
    // inclusive Hillis–Steele scan over 64 slots (rows past c_len scan zeros,
    // so their prefix is the last real G — never read, because every consumer
    // guards on c_len).
    if (tid < DNP_CHUNK) {
        float gv = 0.f;
        float bv = 0.f;
        if (tid < c_len) {
            const size_t row = (size_t)(t0 + tid) * n_v_heads + h;
            gv = a_neg[h] * dn_softplus(alpha[row] + dt_bias[h]);
            bv = dn_sigmoid(blin[row]);
        }
        sg[tid] = gv;
        sb[tid] = bv;
    }
    __syncthreads();
    for (int off = 1; off < DNP_CHUNK; off <<= 1) {
        float add = 0.f;
        if (tid < DNP_CHUNK && tid >= off) add = sg[tid - off];
        __syncthreads();
        if (tid < DNP_CHUNK) sg[tid] += add;
        __syncthreads();
    }
    if (tid < c_len) g_cs[(size_t)h * t_tran + tran + (t0 + tid)] = sg[tid];

    // Stage this K head's k and q rows straight from the Q|K stack — q scaled
    // by the read scale on load, so no scaled copy of q exists anywhere.
    for (int idx = tid; idx < c_len * DNP_DIM; idx += DNP_THREADS) {
        const int i = idx / DNP_DIM;
        const int d = idx % DNP_DIM;
        const float* row = qk + (size_t)(t0 + i) * qk_stride;
        sq[i * DNP_LD + d] = row[kh * DNP_DIM + d] * q_scale;
        sk[i * DNP_LD + d] = row[(n_k_heads + kh) * DNP_DIM + d];
    }
    __syncthreads();

    // A (strict lower) and kq (inclusive), one pass over the (i, j) grid in
    // 4-wide j-tiles — register reuse against the smem bandwidth ceiling: one
    // k_i/q_i operand pair feeds eight FMAs. The upper half is skipped (whole
    // tiles) or discarded at the store (straddling tiles): A's is never read,
    // and kq's is never read because the state pass sums s ≤ t only.
    for (int p = tid; p < DNP_CHUNK * (DNP_CHUNK / 4); p += DNP_THREADS) {
        const int i = p / (DNP_CHUNK / 4);
        const int jt = (p % (DNP_CHUNK / 4)) * 4;
        if (jt > i || i >= c_len) continue;
        const float* ki = &sk[i * DNP_LD];
        const float* qi = &sq[i * DNP_LD];
        float dkk[4] = {0.f, 0.f, 0.f, 0.f};
        float dqk[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 8
        for (int d = 0; d < DNP_DIM; ++d) {
            const float kiv = ki[d];
            const float qiv = qi[d];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float kjv = sk[(jt + q) * DNP_LD + d];
                dkk[q] += kiv * kjv;
                dqk[q] += qiv * kjv;
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int j = jt + q;
            if (j > i) break;
            const float dec = expf(fminf(0.f, sg[i] - sg[j]));
            if (j < i) sA[i * DNP_ALD + j] = sb[i] * dkk[q] * dec;
            kq[((size_t)h * t_tran + tran + (t0 + i)) * DNP_CHUNK + j] = dqk[q] * dec;
        }
    }

    // Right-hand sides, one column per thread: 0..D-1 solve for u (βv),
    // D..2D-1 for w (βk ⊙ e^G). v is a strided view into the conv output.
    const int col = tid;
    const bool is_v = col < DNP_DIM;
    const int d = is_v ? col : col - DNP_DIM;
    float xr[DNP_CHUNK];
    #pragma unroll
    for (int i = 0; i < DNP_CHUNK; ++i) {
        float b = 0.f;
        if (i < c_len) {
            b = is_v
                ? sb[i] * v[(size_t)(t0 + i) * tok_stride + (size_t)h * DNP_DIM + d]
                : sb[i] * sk[i * DNP_LD + d] * expf(sg[i]);
        }
        xr[i] = b;
    }
    __syncthreads(); // A complete before the substitution reads it

    // (I + A) x = b  →  x[i] = b[i] − Σ_{j<i} A[i][j] x[j].
    #pragma unroll
    for (int i = 1; i < DNP_CHUNK; ++i) {
        float acc = xr[i];
        #pragma unroll
        for (int j = 0; j < i; ++j) {
            acc -= sA[i * DNP_ALD + j] * xr[j];
        }
        xr[i] = acc;
    }

    for (int i = 0; i < c_len; ++i) {
        const size_t dst = ((size_t)h * t_tran + tran + (t0 + i)) * DNP_DIM + d;
        if (is_v) u[dst] = xr[i];
        else      w[dst] = xr[i];
    }
}

// ============================================================================
// State pass: one block per (V head, d_v tile of DNP_TV rows), sequential over
// chunks. The block's S tile lives in SMEM in the STORED orientation
// [d_v, d_k] — no s_fla transpose exists anywhere.
//
// The tile was register-distributed in the first version — thread (r, part)
// owning 32 columns of one row — and that layout made every output a 32-FMA
// fragment plus two warp shuffles inside a serial t-loop: dependency chains
// with 4 warps/block and nothing to hide them. Measured 14% achieved warp
// occupancy, 34% SM throughput, 83% of the whole scan's time. With S in smem
// the mapping flips to warp → t, lane → r: each output is one long
// independent dot — stage[t][j] broadcasts across the warp, s_tile[r][j] is
// bank-clean because lanes hold consecutive r on a padded stride — with no
// shuffles, no cross-lane dependencies, 8 warps/block, and coalesced o/vnew
// writes (lanes hold consecutive r). Every dot is 4-way register-tiled (one
// smem operand feeds four FMAs) because the LSU, not the FMA pipe, was the
// measured ceiling.
//
// Dynamic smem (~42 KB — the size that fits TWO blocks per SM, keeping all
// n_v_heads·4 blocks of the grid resident in one wave): s_tile [TV][LD],
// stage [TH][LD] staging HALF a chunk at a time, reused w → q → k,
// vnew [C][TV+1], and the chunk's decay vectors.
// ============================================================================
static __global__ void delta_net_prefill_state_f32_kernel(
        const float* __restrict__ qk_wave,// Q|K columns of the conv output
        const float* __restrict__ u,      // [h_v, T_tran, D]
        const float* __restrict__ w,      // [h_v, T_tran, D]
        const float* __restrict__ kq,     // [h_v, T_tran, C]
        const float* __restrict__ g_cs,   // [h_v, T_tran]
        float*       __restrict__ o_wave, // [T_wave, h_v·D]
        const long long*    __restrict__ ptrs,
        const unsigned int* __restrict__ spans,
        int n_spans,
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride, // conv_dim: q and k are strided views of the conv output
        float q_scale) {
    extern __shared__ float smem[];
    float* s_tile = smem;                            // [TV][LD]
    float* stage = s_tile + DNP_TV * DNP_LD;         // [TH][LD]
    float* vnew  = stage + DNP_TH * DNP_LD;          // [C][TV+1]
    float* sge   = vnew + DNP_CHUNK * (DNP_TV + 1);  // e^{G}
    float* sgd   = sge + DNP_CHUNK;                  // e^{G_last − G}
    __shared__ float s_decay;                        // e^{G_last}

    const DnSpan sp = dn_span(ptrs, spans, n_spans, blockIdx.z);
    const int t_len = sp.len;
    const float* __restrict__ state = sp.state;
    float* __restrict__ state_out = sp.state_out;
    // As in the intra pass: the wave buffers are rebased to the span, and the
    // transients are one wave-wide allocation this span occupies `t_tran`-strided
    // rows of.
    const float* __restrict__ qk = qk_wave + (size_t)sp.start * tok_stride;
    float* __restrict__ o = o_wave + (size_t)sp.start * (size_t)n_v_heads * DNP_DIM;
    const size_t tran = (size_t)sp.start;

    const int h = blockIdx.x;
    const int kh = h % n_k_heads;
    const int i_base = (int)blockIdx.y * DNP_TV;
    const int tid = (int)threadIdx.x; // 256
    const int warp = tid >> 5;        // 0..7
    const int lane = tid & 31;        // = r for the dot phases
    const int qk_stride = tok_stride;
    const size_t o_stride = (size_t)n_v_heads * DNP_DIM;

    // Load the tile: [TV, D] rows are contiguous in global, staged onto the
    // padded stride.
    for (int idx = tid; idx < DNP_TV * DNP_DIM; idx += (int)blockDim.x) {
        const int r = idx / DNP_DIM;
        const int d = idx % DNP_DIM;
        s_tile[r * DNP_LD + d] =
            state[((size_t)h * DNP_DIM + (i_base + r)) * DNP_DIM + d];
    }

    const int n_chunks = (t_len + DNP_CHUNK - 1) / DNP_CHUNK;
    for (int n = 0; n < n_chunks; ++n) {
        const int t0 = n * DNP_CHUNK;
        const int c_len = min(DNP_CHUNK, t_len - t0);

        __syncthreads(); // previous chunk's phase-3 reads are complete
        if (tid == 0) {
            s_decay = expf(g_cs[(size_t)h * t_tran + tran + (t0 + c_len - 1)]);
        }
        if (tid < c_len) {
            const float gv = g_cs[(size_t)h * t_tran + tran + (t0 + tid)];
            const float gl = g_cs[(size_t)h * t_tran + tran + (t0 + c_len - 1)];
            sge[tid] = expf(gv);
            sgd[tid] = expf(gl - gv); // G decreases, so the exponent is ≤ 0
        }

        // Each phase stages HALF a chunk at a time (DNP_TH = 32 tokens): the
        // half-size stage buffer is what fits two blocks on an SM, and two
        // resident blocks are what keep all 128 blocks of the grid in one
        // wave with 16 warps/SM — the occupancy this kernel's smem footprint
        // was costing. The extra syncs per chunk are noise against that.

        // ---- phase 1: stage w; v_new = u − w·Sᵀ (block-local) ----
        for (int hb = 0; hb < c_len; hb += DNP_TH) {
            const int hlen = min(DNP_TH, c_len - hb);
            for (int idx = tid; idx < hlen * DNP_DIM; idx += (int)blockDim.x) {
                const int i = idx / DNP_DIM;
                const int d = idx % DNP_DIM;
                stage[i * DNP_LD + d] =
                    w[((size_t)h * t_tran + tran + (t0 + hb + i)) * DNP_DIM + d];
            }
            __syncthreads();
            // warp → t (stride 8, 4 at a time), lane → r: stage[t][j]
            // broadcasts, s_tile[r][j] hits distinct banks per lane, u/vnew
            // accesses are consecutive in r. The 4-way t-tile is register
            // reuse against the smem bandwidth ceiling: one s_tile operand
            // load feeds four FMAs.
            {
                const float* srow = &s_tile[lane * DNP_LD];
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll 8
                for (int j = 0; j < DNP_DIM; ++j) {
                    const float sv = srow[j];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int t = warp + q * 8;
                        if (t < hlen) acc[q] += stage[t * DNP_LD + j] * sv;
                    }
                }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int t = warp + q * 8;
                    if (t < hlen) {
                        const float uv = u[((size_t)h * t_tran + tran + (t0 + hb + t)) * DNP_DIM +
                                           (i_base + lane)];
                        vnew[(hb + t) * (DNP_TV + 1) + lane] = uv - acc[q];
                    }
                }
            }
            __syncthreads(); // stage free for the next half
        }

        // ---- phase 2: stage q (scaled); o = e^G·(q·Sᵀ) + Σ_{s≤t} kq·v_new ----
        for (int hb = 0; hb < c_len; hb += DNP_TH) {
            const int hlen = min(DNP_TH, c_len - hb);
            for (int idx = tid; idx < hlen * DNP_DIM; idx += (int)blockDim.x) {
                const int i = idx / DNP_DIM;
                const int d = idx % DNP_DIM;
                stage[i * DNP_LD + d] =
                    qk[(size_t)(t0 + hb + i) * qk_stride + kh * DNP_DIM + d] * q_scale;
            }
            __syncthreads();
            {
                const float* srow = &s_tile[lane * DNP_LD];
                float inter[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll 8
                for (int j = 0; j < DNP_DIM; ++j) {
                    const float sv = srow[j];
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int t = warp + q * 8;
                        if (t < hlen) inter[q] += stage[t * DNP_LD + j] * sv;
                    }
                }
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int t = warp + q * 8;
                    if (t >= hlen) continue;
                    const int tc = hb + t; // within-chunk token index
                    // The intra-chunk read of this chunk's own writes: kq
                    // rows broadcast across the warp, vnew (fully written in
                    // phase 1) is bank-clean per lane.
                    const float* kqrow =
                        kq + ((size_t)h * t_tran + tran + (t0 + tc)) * DNP_CHUNK;
                    float intra = 0.f;
                    for (int s = 0; s <= tc; ++s) {
                        intra += __ldg(kqrow + s) * vnew[s * (DNP_TV + 1) + lane];
                    }
                    o[(size_t)(t0 + tc) * o_stride + (size_t)h * DNP_DIM +
                      (i_base + lane)] =
                        inter[q] * sge[tc] + intra; // pre-update S: inter-chunk read
                }
            }
            __syncthreads(); // stage free
        }

        // ---- phase 3: stage k; S ← e^{G_last}·S + v_newᵀ(k ⊙ e^{G_last−G}) ----
        // Each thread owns 16 S elements: j fixed per half-block, r striped —
        // disjoint (r, j) pairs, so the in-place update has no races. All 16
        // accumulators live in registers (persisting across the staged
        // halves) so one stage[t][j]·sgd[t] operand load feeds 16 FMAs.
        {
            const int j = tid & (DNP_DIM - 1);        // 0..127
            const int r0 = (tid >> 7) * (DNP_TV / 2); // 0 or 16
            float acc[DNP_TV / 2];
            #pragma unroll
            for (int rr = 0; rr < DNP_TV / 2; ++rr) acc[rr] = 0.f;
            for (int hb = 0; hb < c_len; hb += DNP_TH) {
                const int hlen = min(DNP_TH, c_len - hb);
                for (int idx = tid; idx < hlen * DNP_DIM; idx += (int)blockDim.x) {
                    const int i = idx / DNP_DIM;
                    const int d = idx % DNP_DIM;
                    stage[i * DNP_LD + d] =
                        qk[(size_t)(t0 + hb + i) * qk_stride +
                           (n_k_heads + kh) * DNP_DIM + d];
                }
                __syncthreads();
                for (int t = 0; t < hlen; ++t) {
                    const float kg = stage[t * DNP_LD + j] * sgd[hb + t];
                    const float* vrow = &vnew[(hb + t) * (DNP_TV + 1) + r0];
                    #pragma unroll
                    for (int rr = 0; rr < DNP_TV / 2; ++rr) {
                        acc[rr] += vrow[rr] * kg;
                    }
                }
                __syncthreads(); // stage free for the next half
            }
            const float dec = s_decay;
            #pragma unroll
            for (int rr = 0; rr < DNP_TV / 2; ++rr) {
                float* sp = &s_tile[(r0 + rr) * DNP_LD + j];
                *sp = *sp * dec + acc[rr];
            }
        }
    }

    // The advanced tile goes to `state_out`, which the wave points at the slot's
    // OTHER buffer. Every element this block loaded is written back, and the grid
    // covers every (head, d_v-tile), so the destination is fully written and
    // carries nothing forward from whatever it last held — which is what lets a
    // failed wave roll back by not swapping the two buffers rather than by
    // copying the entering state aside first. `state_out == state` is also legal
    // (the reference path passes one buffer twice): the tile is already resident
    // in shared memory by the time it is stored.
    __syncthreads();
    for (int idx = tid; idx < DNP_TV * DNP_DIM; idx += (int)blockDim.x) {
        const int r = idx / DNP_DIM;
        const int d = idx % DNP_DIM;
        state_out[((size_t)h * DNP_DIM + (i_base + r)) * DNP_DIM + d] =
            s_tile[r * DNP_LD + d];
    }
}

static inline void launch_conv_prefill_f32(
        const float* x_wave,
        const float* kernel,
        float* y_wave,
        const long long* ptrs,
        const unsigned int* spans,
        int n_spans,
        int max_len,
        int channels,
        int kwidth,
        int qk_channels,
        float eps,
        cudaStream_t stream) {
    if (n_spans <= 0 || max_len <= 0 || channels <= 0 || kwidth <= 1) return;
    // The epilogue's norm reduction is block-local; a block must hold whole
    // head groups, which qk_channels = h_k·256 guarantees at 256 threads.
    if (qk_channels < 0 || qk_channels > channels || qk_channels % 256 != 0) return;
    const int threads = 256;
    dim3 grid((channels + threads - 1) / threads, max_len, n_spans);
    delta_net_conv_prefill_f32_kernel<<<grid, threads, 0, stream>>>(
        x_wave, kernel, y_wave, ptrs, spans, n_spans, channels, kwidth,
        qk_channels, eps);
}

static inline void launch_prefill_intra_f32(
        const float* qk_wave,
        const float* v_wave,
        const float* alpha_wave,
        const float* blin_wave,
        const float* dt_bias,
        const float* a_neg,
        float* u,
        float* w,
        float* kq,
        float* g_cs,
        const unsigned int* spans,
        int n_spans,
        int max_len,
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        cudaStream_t stream) {
    if (n_spans <= 0 || max_len <= 0 || n_v_heads <= 0 || n_k_heads <= 0) return;
    const int smem_bytes =
        (2 * DNP_CHUNK * DNP_LD + DNP_CHUNK * DNP_ALD + 2 * DNP_CHUNK) *
        (int)sizeof(float);
    // 83.5 KB exceeds the 48 KB default dynamic-smem ceiling; raise it once.
    // The attribute is per-function and sticky, so a redundant set is a no-op.
    static int smem_raised = 0;
    if (!smem_raised) {
        cudaFuncSetAttribute(delta_net_prefill_intra_f32_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        smem_raised = 1;
    }
    // Chunks for the WIDEST span: shorter spans' surplus blocks return at the
    // top of the kernel. A rectangle wastes at most `max_len − len` block rows
    // per span, which is nothing against the launch it replaces.
    const int n_chunks = (max_len + DNP_CHUNK - 1) / DNP_CHUNK;
    dim3 grid(n_chunks, n_v_heads, n_spans);
    delta_net_prefill_intra_f32_kernel<<<grid, DNP_THREADS, smem_bytes, stream>>>(
        qk_wave, v_wave, alpha_wave, blin_wave, dt_bias, a_neg, u, w, kq, g_cs,
        spans, n_spans, t_tran, n_v_heads, n_k_heads, tok_stride, q_scale);
}

static inline void launch_prefill_state_f32(
        const float* qk_wave,
        const float* u,
        const float* w,
        const float* kq,
        const float* g_cs,
        float* o_wave,
        const long long* ptrs,
        const unsigned int* spans,
        int n_spans,
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        cudaStream_t stream) {
    if (n_spans <= 0 || n_v_heads <= 0 || n_k_heads <= 0) return;
    // ~42 KB — deliberately under the 48 KB default so two blocks share an SM.
    const int smem_bytes = (DNP_TV * DNP_LD + DNP_TH * DNP_LD +
                            DNP_CHUNK * (DNP_TV + 1) + 2 * DNP_CHUNK) *
                           (int)sizeof(float);
    // No span dimension in the grid's extent beyond `n_spans`: this pass walks
    // its span's chunks serially inside the block, so its shape never depended
    // on the length.
    dim3 grid(n_v_heads, DNP_DIM / DNP_TV, n_spans);
    delta_net_prefill_state_f32_kernel<<<grid, 256, smem_bytes, stream>>>(
        qk_wave, u, w, kq, g_cs, o_wave, ptrs, spans, n_spans, t_tran,
        n_v_heads, n_k_heads, tok_stride, q_scale);
}

} // namespace delta_net
