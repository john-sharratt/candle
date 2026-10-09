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
// kernels keep it on chip (in shared memory) across the whole sequence.
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
//   conv_prefill — causal conv over 256 channels × DNC_TOK tokens a block.
//   intra        — per (V-head, chunk): gates, G scan, A build, one forward-
//                  substitution solve for both right-hand sides, and the
//                  inclusive dot grid kq.
//   state        — per (V-head, d_v-tile): the sequential chunk walk with the
//                  S tile resident in shared memory in the stored orientation
//                  and the output fused.
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
// d_v rows of state owned by one state-pass block.
#define DNP_TV 32
// The short-span kernel's padded row strides: multiples of 4 floats, so every
// row is 16-byte aligned and its dots read four columns per load. +4 rather
// than +1 keeps a quarter-warp's lane-distinct float4 rows on distinct banks.
// (The state pass swizzles instead of padding — see its header.)
#define DNP_SLD (DNP_DIM + 4)
#define DNP_VLD (DNP_TV + 4)
// Tokens the state pass stages per half-chunk — half of DNP_CHUNK, so its
// stage buffer is half-size and two blocks fit an SM (see the kernel header).
#define DNP_TH 32
// float4 columns of a D-wide row, and of a kq row.
#define DNP_F4 (DNP_DIM / 4)
#define DNP_KQ_F4 (DNP_CHUNK / 4)
// float4s each state-pass thread moves per staged half (DNP_TH rows of D) —
// equally, per state-tile load or store (DNP_TV rows of D) — and per half of
// kq rows (DNP_TH rows of C).
#define DNP_STAGE_F4 (DNP_TH * DNP_F4 / DNP_THREADS)
#define DNP_KQ_F4_PER_THREAD (DNP_TH * DNP_KQ_F4 / DNP_THREADS)
// S rows each state-pass thread updates, for each of its two columns.
#define DNP_UPD_ROWS (DNP_TV * DNP_DIM / (2 * DNP_THREADS))
// Warps of a state-pass block that run the dot phases (4 tokens × 2 rows a
// lane, so four warps cover a staged half's DNP_TH × DNP_TV outputs).
#define DNP_DOT_WARPS (DNP_TH * DNP_TV / (8 * 32))
// The state pass's dynamic shared memory: s_tile, stage, skq, vnew, e^G and
// e^{G_last−G}.
#define DNP_STATE_SMEM_FLOATS \
    (DNP_TV * DNP_DIM + DNP_TH * DNP_DIM + DNP_TH * DNP_CHUNK + DNP_CHUNK * DNP_TV + 2 * DNP_CHUNK)

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

// ============================================================================
// The layer stack — a speculative rewind's form of the same launch.
//
// A rewind replays the accepted prefix through EVERY recurrent layer, and no
// layer's replay reads another's output: each starts from its own entering
// state and its own stashed operands. So the layers are as independent as the
// spans are, and they take the same answer — one launch for the stack, with
// `blockIdx.z = layer · n_spans + span`.
//
// What differs per layer comes in two kinds:
//   * operands that live in their own allocations (the stashed projections,
//     the layer's constants) — read from a `DnLayerOps` table;
//   * buffers the launch itself carves (the conv output, the scan transients,
//     the discarded output, the span pointer rows) — stacked by layer in one
//     allocation each, so the layer's slice is a fixed stride from the base.
//
// `layers == nullptr` is the single-layer launch the forward makes: every
// pointer argument is the layer's own and the layer index is 0. One kernel
// serves both, so a rewind retraces the wave's arithmetic by construction.
// ============================================================================
struct DnLayerOps {
    const float* x;       // the conv's raw input rows [T, C]
    const float* kernel;  // the conv weights [C, K]
    const float* alpha;   // raw decay-gate projection [T, h_v]
    const float* blin;    // raw beta projection [T, h_v]
    const float* dt_bias; // [h_v]
    const float* a_neg;   // [h_v]
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
// `x`/`y` are the whole packed buffers rebased by its `start`. With a layer
// table, `blockIdx.z` picks the (layer, span) pair and `y`/`ptrs` are the
// layer-stacked buffers.
//
// A block is 256 channels × DNC_TOK consecutive tokens. Each thread holds its
// channel's window of DNC_TOK + K − 1 inputs in registers, so an input row is
// read by ~1.2 blocks rather than K, every load is issued before any
// arithmetic, and the Q|K blocks run the l2-norm tree for all their tokens
// together: the two 128→32 levels through shared memory, the last five by warp
// shuffles — slot l adds slot l + off at every level, the tree
// dn_silu_norm_epilogue walks, so the root is the same sum (the short-span
// kernel reduces the same way). Each output is the same j-ascending FMA chain
// from 0 as one thread per (token, channel) computed it.
// ============================================================================
// Tokens per conv block, and the widest conv window it takes.
#define DNC_TOK 16
#define DNC_KMAX 8

static __global__ void __launch_bounds__(256, 4) delta_net_conv_prefill_f32_kernel(
        const float* __restrict__ x_wave,  // [T_wave, C]
        const float* __restrict__ kernel,  // [C, K]
        float*       __restrict__ y_wave,  // [layers, T_wave, C]
        const long long*    __restrict__ ptrs,   // [layers, 4, n_spans]
        const unsigned int* __restrict__ spans,
        int n_spans,
        const DnLayerOps* __restrict__ layers,   // null: one layer, the args
        int t_wave,
        int channels,
        int kwidth,
        int qk_channels,
        float eps) {
    __shared__ float red[DNC_TOK * 256];
    __shared__ float snorm[DNC_TOK * 2];
    const int tid = (int)threadIdx.x;
    const int c = blockIdx.x * 256 + tid;
    const bool live = c < channels;
    const int layer = (int)blockIdx.z / n_spans;
    const int z = (int)blockIdx.z - layer * n_spans;
    if (layers != nullptr) {
        x_wave = layers[layer].x;
        kernel = layers[layer].kernel;
        y_wave += (size_t)layer * t_wave * channels;
        ptrs += (size_t)layer * 4 * n_spans;
    }
    const DnSpan sp = dn_span(ptrs, spans, n_spans, z);
    const int tb0 = (int)blockIdx.y * DNC_TOK;
    // The launch is a rectangle over the WIDEST span, so shorter spans leave
    // block rows with no token. `tb0` is per block row, so this is uniform
    // across the block and the norm reduction below is never entered by only
    // part of a block.
    if (tb0 >= sp.len) return;
    const float* __restrict__ x = x_wave + (size_t)sp.start * channels;
    float* __restrict__ y = y_wave + (size_t)sp.start * channels;
    const float* __restrict__ tail = sp.tail;
    float* __restrict__ tail_out = sp.tail_out;
    const int t_len = sp.len;
    const int tcols = kwidth - 1;
    const int n_tok = min(DNC_TOK, t_len - tb0);

    // Weights, then the window: win[p] = in(tb0 − (K−1) + p), where in(i) is
    // x for 0 ≤ i < len and the entering tail for i < 0.
    float kw_r[DNC_KMAX];
    #pragma unroll
    for (int j = 0; j < DNC_KMAX; ++j) {
        kw_r[j] = (live && j < kwidth) ? kernel[(size_t)c * kwidth + j] : 0.f;
    }
    float win[DNC_TOK + DNC_KMAX - 1];
    #pragma unroll
    for (int p = 0; p < DNC_TOK + DNC_KMAX - 1; ++p) {
        const int idx = tb0 - tcols + p;
        win[p] = 0.f;
        if (live && p < DNC_TOK + tcols && idx < t_len) {
            win[p] = (idx >= 0) ? x[(size_t)idx * channels + c]
                                : tail[(size_t)c * tcols + (tcols + idx)];
        }
    }
    float sv[DNC_TOK];
    #pragma unroll
    for (int tt = 0; tt < DNC_TOK; ++tt) {
        float acc = 0.f;
        #pragma unroll
        for (int j = 0; j < DNC_KMAX; ++j) {
            if (j < kwidth) acc += kw_r[j] * win[tt + j];
        }
        sv[tt] = dn_silu(acc);
    }

    // Block-uniform: qk_channels is a multiple of 256, so a block is all Q|K
    // channels or none.
    if ((int)blockIdx.x * 256 < qk_channels) {
        #pragma unroll
        for (int tt = 0; tt < DNC_TOK; ++tt) red[tt * 256 + tid] = sv[tt] * sv[tt];
        __syncthreads();
        if ((tid & (DN_HEAD_DIM - 1)) < DN_HEAD_DIM / 2) {
            #pragma unroll
            for (int tt = 0; tt < DNC_TOK; ++tt) {
                if (tt < n_tok) red[tt * 256 + tid] += red[tt * 256 + tid + DN_HEAD_DIM / 2];
            }
        }
        __syncthreads();
        if ((tid & (DN_HEAD_DIM - 1)) < 32) {
            #pragma unroll
            for (int tt = 0; tt < DNC_TOK; ++tt) {
                if (tt < n_tok) {
                    float v = red[tt * 256 + tid] + red[tt * 256 + tid + 32];
                    #pragma unroll
                    for (int off = 16; off >= 1; off >>= 1) {
                        v += __shfl_down_sync(0xffffffffu, v, off);
                    }
                    if ((tid & 31) == 0) snorm[tt * 2 + (tid >> 7)] = v;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int tt = 0; tt < DNC_TOK; ++tt) {
            if (tt < n_tok) {
                y[(size_t)(tb0 + tt) * channels + c] =
                    dn_l2_scale(sv[tt], snorm[tt * 2 + (tid >> 7)], eps);
            }
        }
    } else if (live) {
        #pragma unroll
        for (int tt = 0; tt < DNC_TOK; ++tt) {
            if (tt < n_tok) y[(size_t)(tb0 + tt) * channels + c] = sv[tt];
        }
    }

    // One block row owns the tail write; its reads see only the old buffer.
    if (blockIdx.y == 0 && live) {
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
// Dynamic smem partition: sk [R][D], sq [R][D], A [R][DNI_ALD], G [C], β [C],
// where R = `rows` is the launch's chunk-row capacity — DNP_CHUNK (81.5 KiB, the
// launcher opts in past the 48 KB default) when any span fills a chunk, and the
// longest span's length rounded up to 4 otherwise (the A/kq grid reads k rows a
// whole 4-wide j-tile at a time). A verify wave's spans are a few tokens long, and
// sizing them for a 64-token chunk held the kernel to one block per SM for rows it
// never touches. G and β stay chunk-wide: the scan walks all DNP_CHUNK slots.
// sk and sq are unpadded and XOR-swizzled at float4 granularity by the row's
// 4-row group (dni_swz), which keeps the A/kq grid's j-tile rows on distinct
// banks; A's rows are 16-byte aligned so the solve reads four entries a load.
//
// The kernel runs one block per SM (the row buffers are 81.5 KiB), so it is built
// for latency, not occupancy:
//   * every global load is issued before the first shared store — the q/k rows
//     (four float4 per row-thread) and each u-column thread's 64 v values — so
//     their DRAM latency is paid once, behind the gate scan;
//   * the A/kq grid is register-tiled 2 rows × 4 columns, every tile below or
//     on the diagonal enumerated once (272 at a full chunk, one pass of the
//     block and a sixteenth), four d per shared load: eight loads feed 64 FMAs;
//   * the substitution walks rows in pairs, so two independent chains are in
//     flight where one serial chain stalled on every FMA's latency.
//
// The solve assigns one right-hand-side column per thread: d_v + d_k = 256
// columns = 256 threads exactly. X lives in registers with the 64-step
// substitution fully unrolled (static indices — a dynamic index would put the
// array in local memory, the store loop included); A[i][j] reads at step i are
// the same row for every thread, i.e. smem broadcasts. Rows at and past c_len
// hold garbage in both A and X; they are never stored, and clean rows
// i < c_len only ever read x[j] for j < i, so the garbage stays confined to
// discarded lanes.
//
// **Bit-identical to the scalar form it replaces**: each A/kq entry is one
// d-ascending FMA chain, each substitution row subtracts its terms in j order
// from its own right-hand side, and the gates, the scan and the right-hand
// sides are the same expressions. `prefill_launches_reproduce_the_recorded_bits`
// holds it.
// ============================================================================
// A's row stride: 16-byte aligned rows.
#define DNI_ALD (DNP_CHUNK + 4)

// Slot of float4 column `c4` of row `row` in sk / sq.
static __device__ __forceinline__ int dni_swz(int row, int c4) {
    return c4 ^ ((row >> 2) & 7);
}

static __global__ void __launch_bounds__(DNP_THREADS, 1) delta_net_prefill_intra_f32_kernel(
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
        const DnLayerOps* __restrict__ layers, // null: one layer, the args
        int t_tran,     // rows per head of the shared transients, and rows of
                        // the conv output a layer's slice holds
        int n_v_heads,
        int n_k_heads,
        int tok_stride, // conv_dim: q, k and v are strided views of one buffer
        float q_scale,
        int rows) {     // chunk rows the row buffers hold — every span's c_len,
                        // rounded up to the A/kq grid's 4-wide j-tile
    extern __shared__ __align__(16) float smem[];
    float* sk = smem;                          // [R][D]   swizzled
    float* sq = sk + rows * DNP_DIM;           // [R][D]   swizzled
    float* sA = sq + rows * DNP_DIM;           // [R][DNI_ALD]
    float* sg = sA + rows * DNI_ALD;           // [C] G cumsum
    float* sb = sg + DNP_CHUNK;                // [C] β
    float4* sk4 = reinterpret_cast<float4*>(sk);
    float4* sq4 = reinterpret_cast<float4*>(sq);

    const int layer = (int)blockIdx.z / n_spans;
    const int z = (int)blockIdx.z - layer * n_spans;
    if (layers != nullptr) {
        const size_t conv_slice = (size_t)layer * t_tran * tok_stride;
        const size_t head_rows = (size_t)layer * n_v_heads * t_tran;
        qk_wave += conv_slice;
        v_wave += conv_slice;
        alpha_wave = layers[layer].alpha;
        blin_wave = layers[layer].blin;
        dt_bias = layers[layer].dt_bias;
        a_neg = layers[layer].a_neg;
        u += head_rows * DNP_DIM;
        w += head_rows * DNP_DIM;
        kq += head_rows * DNP_CHUNK;
        g_cs += head_rows;
    }
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
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t qk_stride = (size_t)tok_stride;

    // ---- every global load first ------------------------------------------
    // This K head's q and k rows: float4 column `lane` of rows warp, warp + 8, …
    // (a warp reads a whole 512-byte row per load).
    float4 qreg[DNP_CHUNK / 8];
    float4 kreg[DNP_CHUNK / 8];
    #pragma unroll
    for (int k = 0; k < DNP_CHUNK / 8; ++k) {
        const int i = warp + 8 * k;
        if (i < c_len) {
            const float* row = qk + (size_t)(t0 + i) * qk_stride;
            qreg[k] = *reinterpret_cast<const float4*>(row + kh * DNP_DIM + 4 * lane);
            kreg[k] = *reinterpret_cast<const float4*>(row + (n_k_heads + kh) * DNP_DIM +
                                                       4 * lane);
        }
    }
    // The solve's right-hand-side columns: 0..D-1 solve for u (βv), D..2D-1 for
    // w (βk ⊙ e^G). A u column's v values are loaded raw now and weighted by β
    // once the gates exist; v is a strided view into the conv output.
    const int col = tid;
    const bool is_v = col < DNP_DIM;
    const int d = is_v ? col : col - DNP_DIM;
    float xr[DNP_CHUNK];
    #pragma unroll
    for (int i = 0; i < DNP_CHUNK; ++i) {
        xr[i] = 0.f;
        if (is_v && i < c_len) xr[i] = v[(size_t)(t0 + i) * tok_stride + (size_t)h * DNP_DIM + d];
    }

    // Gates for the chunk (computed here, not by the caller), then an
    // inclusive Hillis–Steele scan over 64 slots (rows past c_len scan zeros,
    // so their prefix is the last real G — never read, because every consumer
    // guards on c_len).
    if (tid < DNP_CHUNK) {
        float gv = 0.f;
        float bv = 0.f;
        if (tid < c_len) {
            const size_t row = (size_t)(t0 + tid) * n_v_heads + h;
            gv = dn_decay_gate(a_neg[h], alpha[row], dt_bias[h]);
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

    // Stage this K head's k and q rows — q scaled by the read scale on the way
    // in, so no scaled copy of q exists anywhere.
    #pragma unroll
    for (int k = 0; k < DNP_CHUNK / 8; ++k) {
        const int i = warp + 8 * k;
        if (i < c_len) {
            float4 q = qreg[k];
            q.x = q.x * q_scale;
            q.y = q.y * q_scale;
            q.z = q.z * q_scale;
            q.w = q.w * q_scale;
            sq4[i * DNP_F4 + dni_swz(i, lane)] = q;
            sk4[i * DNP_F4 + dni_swz(i, lane)] = kreg[k];
        }
    }
    __syncthreads();

    // A (strict lower) and kq (inclusive) over the (i, j) grid in 2×4 tiles:
    // rows {2·ip, 2·ip + 1} × columns 4·jq..4·jq+3 for every jq ≤ ip / 2 — the
    // tiles that hold an entry on or below the diagonal, enumerated row-pair by
    // row-pair (row pair ip opens at tile ⌊ip/2⌋·(⌊ip/2⌋+1) + (ip odd ? ⌊ip/2⌋+1 : 0)).
    // A straddling tile's upper entries are discarded at the store: A's upper
    // half is never read, and kq's is never read because the state pass sums
    // s ≤ t only.
    {
        const int n_ip = (c_len + 1) >> 1;
        const int n_tiles = (n_ip >> 1) * ((n_ip >> 1) + 1) + ((n_ip & 1) ? (n_ip >> 1) + 1 : 0);
        for (int f = tid; f < n_tiles; f += DNP_THREADS) {
            // Invert the enumeration: m = ⌊ip/2⌋ is the largest m with
            // m·(m+1) ≤ f, then the odd row pair of m starts at (m+1)².
            int m = (int)((sqrtf(4.f * (float)f + 1.f) - 1.f) * 0.5f);
            while ((m + 1) * (m + 2) <= f) ++m;
            while (m * (m + 1) > f) --m;
            int ip, jq;
            if (f >= (m + 1) * (m + 1)) {
                ip = 2 * m + 1;
                jq = f - (m + 1) * (m + 1);
            } else {
                ip = 2 * m;
                jq = f - m * (m + 1);
            }
            const int i0 = 2 * ip;
            const int j0 = 4 * jq;
            const int zi = ip >> 1;     // (i0 >> 2) & 7, before the mask
            const int zj = jq & 7;      // (j0 >> 2) & 7
            const float4* ki0 = sk4 + i0 * DNP_F4;
            const float4* ki1 = ki0 + DNP_F4;
            const float4* qi0 = sq4 + i0 * DNP_F4;
            const float4* qi1 = qi0 + DNP_F4;
            const float4* kj = sk4 + j0 * DNP_F4;
            float dkk[2][4];
            float dqk[2][4];
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    dkk[a][b] = 0.f;
                    dqk[a][b] = 0.f;
                }
            }
            #pragma unroll 2
            for (int mm = 0; mm < DNP_F4 / 8; ++mm) {
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int si = 8 * mm + (c ^ (zi & 7));
                    const int sj = 8 * mm + (c ^ zj);
                    const float4 ka[2] = {ki0[si], ki1[si]};
                    const float4 qa[2] = {qi0[si], qi1[si]};
                    float4 kb[4];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) kb[b] = kj[b * DNP_F4 + sj];
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            dkk[a][b] += ka[a].x * kb[b].x;
                            dqk[a][b] += qa[a].x * kb[b].x;
                            dkk[a][b] += ka[a].y * kb[b].y;
                            dqk[a][b] += qa[a].y * kb[b].y;
                            dkk[a][b] += ka[a].z * kb[b].z;
                            dqk[a][b] += qa[a].z * kb[b].z;
                            dkk[a][b] += ka[a].w * kb[b].w;
                            dqk[a][b] += qa[a].w * kb[b].w;
                        }
                    }
                }
            }
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int i = i0 + a;
                if (i >= c_len) continue;
                float* kq_row = kq + ((size_t)h * t_tran + tran + (t0 + i)) * DNP_CHUNK;
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const int j = j0 + b;
                    if (j > i) continue;
                    const float dec = expf(fminf(0.f, sg[i] - sg[j]));
                    if (j < i) sA[i * DNI_ALD + j] = sb[i] * dkk[a][b] * dec;
                    kq_row[j] = dqk[a][b] * dec;
                }
            }
        }
    }

    // Right-hand sides: βv for the u columns (v already in xr), βk ⊙ e^G for
    // the w columns.
    #pragma unroll
    for (int i = 0; i < DNP_CHUNK; ++i) {
        if (i < c_len) {
            xr[i] = is_v
                ? sb[i] * xr[i]
                : sb[i] * sk[i * DNP_DIM + 4 * dni_swz(i, d >> 2) + (d & 3)] * expf(sg[i]);
        }
    }
    __syncthreads(); // A complete before the substitution reads it

    // (I + A) x = b  →  x[i] = b[i] − Σ_{j<i} A[i][j] x[j], every row's terms
    // subtracted in j order. Rows go in pairs (i, i + 1): row i + 1's first i
    // terms need only x[0..i−1], so its chain runs beside row i's and takes its
    // last term once x[i] lands.
    //
    // Stops at `c_len` (block-uniform): rows past it are never stored, and a row
    // only reads the rows before it, so the stored ones are the same values — at a
    // verify-width span of four tokens the full walk was ~2,000 FMAs a thread for
    // four rows. The loop stays fully unrolled so `xr` keeps static indices.
    #pragma unroll
    for (int i = 1; i < DNP_CHUNK; i += 2) {
        if (i >= c_len) break;
        const float* a0 = sA + i * DNI_ALD;
        if (i + 1 < DNP_CHUNK) {
            const float* a1 = a0 + DNI_ALD;
            float acc0 = xr[i];
            float acc1 = xr[i + 1];
            #pragma unroll
            for (int j = 0; j < i; ++j) {
                acc0 -= a0[j] * xr[j];
                acc1 -= a1[j] * xr[j];
            }
            xr[i] = acc0;
            acc1 -= a1[i] * xr[i];
            xr[i + 1] = acc1;
        } else {
            float acc0 = xr[i];
            #pragma unroll
            for (int j = 0; j < i; ++j) {
                acc0 -= a0[j] * xr[j];
            }
            xr[i] = acc0;
        }
    }

    float* __restrict__ dst = (is_v ? u : w) + ((size_t)h * t_tran + tran + t0) * DNP_DIM + d;
    #pragma unroll
    for (int i = 0; i < DNP_CHUNK; ++i) {
        if (i < c_len) dst[(size_t)i * DNP_DIM] = xr[i];
    }
}

// ============================================================================
// acc[q] += Σ_j stage[warp + 8q][j] · srow[j] for the warp's (up to) four
// tokens below `hlen`, four columns per shared load, on the padded DNP_SLD
// rows. Each accumulator sums its columns in ascending order — the order of
// every state-pass dot — and the short-span kernel's dots run through it.
// ============================================================================
static __device__ __forceinline__ void dnp_dot4(
        const float* __restrict__ srow,
        const float* __restrict__ stage,
        int warp,
        int hlen,
        float acc[4]) {
    const float4* s4 = reinterpret_cast<const float4*>(srow);
    #pragma unroll 4
    for (int j4 = 0; j4 < DNP_DIM / 4; ++j4) {
        const float4 sv = s4[j4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int t = warp + q * 8;
            if (t < hlen) {
                const float4 st = reinterpret_cast<const float4*>(&stage[t * DNP_SLD])[j4];
                acc[q] += st.x * sv.x;
                acc[q] += st.y * sv.y;
                acc[q] += st.z * sv.z;
                acc[q] += st.w * sv.w;
            }
        }
    }
}

// ============================================================================
// State pass: one block per (V head, d_v tile of DNP_TV rows), sequential over
// chunks. The block's S tile lives in SMEM in the STORED orientation
// [d_v, d_k] — no s_fla transpose exists anywhere.
//
// Each chunk is six steps — w, q and k, each staged half a chunk (DNP_TH
// tokens) at a time — and every step's global operand is loaded into
// registers during the step BEFORE it: a step is barrier, store the operand
// that arrived, barrier, issue the next step's loads, compute. The compute is
// what hides the loads. Staging straight from global into shared (load,
// store, load, store…) put a dependent DRAM round trip behind every element:
// measured 51% of the kernel's stall samples on the stores alone, at 7.8 ms
// per 7.5K-token layer.
//
// With the loads hidden, shared-memory wavefronts are the ceiling, so the
// dots are register-tiled 4×2 (four tokens × two rows per thread, see
// dnp_dot42 for the wavefront arithmetic) and the update 2×8 (two columns ×
// eight rows).
//
// Shared layouts are UNPADDED and XOR-swizzled at float4 granularity —
// float4 column c4 of row `row` sits in slot c4 ^ (row & 7) — so the lanes of
// every access below land on distinct banks without the padding a stride
// would cost. That is what fits the kq rows in the ~48.5 KB that keeps two
// blocks on an SM (all n_v_heads·4 blocks resident in one wave, 16 warps/SM):
//   s_tile [TV][D], stage [TH][D], skq [TH][C], vnew [C][TV], e^G, e^{G_last−G}.
//
// **The arithmetic is unchanged to the bit.** Every output is still the same
// chain: v_new and the inter-chunk read sum their d_k columns 0, 1, 2, … in
// one accumulator, the intra-chunk read sums s = 0..t, the update sums t
// ascending across both halves and lands as S·e^{G_last} + acc; q is scaled
// and k weighted by e^{G_last−G} with the same single multiplies, now applied
// once when staged. Only which thread computes a value and when its operand
// arrives differ. `prefill_launches_reproduce_the_recorded_bits`
// (delta_net/cuda/tests/prefill_golden.rs) holds it.
// ============================================================================

// Slot of float4 column `c4` in row `row` of a swizzled shared matrix.
static __device__ __forceinline__ int dnp_swz(int row, int c4) {
    return c4 ^ (row & 7);
}

static __device__ __forceinline__ void dnp_put_row4(float* stage, int t, int c4, float4 v) {
    reinterpret_cast<float4*>(stage)[t * DNP_F4 + dnp_swz(t, c4)] = v;
}

static __device__ __forceinline__ float dnp_f4_at(const float4& v, int e) {
    return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w;
}

// The thread's share of rows [0, n) of a D-wide operand whose rows are `ld`
// floats apart: float4 column `lane` of rows warp, warp + 8, … — a warp reads
// one whole 512-byte row per load. Rows at or past `n` are not read (their
// registers keep stale values, which nothing consumes).
static __device__ __forceinline__ void dnp_fetch_rows(
        const float* __restrict__ base, size_t ld, int n, int warp, int lane,
        float4 pre[DNP_STAGE_F4]) {
    #pragma unroll
    for (int k = 0; k < DNP_STAGE_F4; ++k) {
        const int t = warp + 8 * k;
        if (t < n) pre[k] = *reinterpret_cast<const float4*>(base + (size_t)t * ld + 4 * lane);
    }
}

// The thread's share of kq rows [0, n): DNP_KQ_F4 float4 per row.
static __device__ __forceinline__ void dnp_fetch_kq(
        const float* __restrict__ base, int n, int tid, float4 pkq[DNP_KQ_F4_PER_THREAD]) {
    #pragma unroll
    for (int k = 0; k < DNP_KQ_F4_PER_THREAD; ++k) {
        const int e = tid + k * DNP_THREADS;
        const int t = e / DNP_KQ_F4;
        if (t < n) {
            pkq[k] = *reinterpret_cast<const float4*>(base + (size_t)t * DNP_CHUNK +
                                                      4 * (e % DNP_KQ_F4));
        }
    }
}

// The state pass's dot tile: for the four tokens t = tq + 8k (k = 0..3) and
// the two rows r ∈ {ra, ra + 8},
//   acc[k][r] += Σ_j stage[t][j] · s_tile[r][j],   j ascending,
// four columns per shared load.
//
// Why 4 × 2 and why this lane map: a 16-byte shared load costs one wavefront
// per quarter-warp (8 lanes) that reads more than one address, and half a
// wavefront per quarter-warp that reads exactly one. A quarter-warp here shares
// its token (tq) and spreads its 8 lanes over 8 rows (ra), so each stage load
// costs 2 wavefronts and each S load 4: 16 per 32 FMAs, where a 2 × 2 tile
// paid 12 per 16. The swizzles: rows ra and ra + 8 share `ra & 7`, and the
// four token rows share `tq & 7`, each pattern landing on distinct banks.
static __device__ __forceinline__ void dnp_dot42(
        const float4* __restrict__ s4, const float4* __restrict__ st4, int ra, int tq,
        float acc[4][2]) {
    const float4* sa = s4 + ra * DNP_F4;
    const float4* sb = sa + 8 * DNP_F4;
    const float4* xr = st4 + tq * DNP_F4;
    const int zr = ra & 7;
    const int zt = tq & 7;
    #pragma unroll
    for (int m = 0; m < DNP_F4 / 8; ++m) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float4 s_a = sa[8 * m + (c ^ zr)];
            const float4 s_b = sb[8 * m + (c ^ zr)];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float4 x = xr[k * 8 * DNP_F4 + 8 * m + (c ^ zt)];
                acc[k][0] += x.x * s_a.x;
                acc[k][0] += x.y * s_a.y;
                acc[k][0] += x.z * s_a.z;
                acc[k][0] += x.w * s_a.w;
                acc[k][1] += x.x * s_b.x;
                acc[k][1] += x.y * s_b.y;
                acc[k][1] += x.z * s_b.z;
                acc[k][1] += x.w * s_b.w;
            }
        }
    }
}

// The thread maps the state pass is written against.
static_assert(DNP_THREADS == 256, "eight warps: four dot warps of 16 tokens × 16 rows");
static_assert(DNP_TV == 32 && DNP_TH == 32, "four dot warps tile a 32 × 32 half");
static_assert(DNP_DIM == 128, "the update owns columns ja and ja + 64");
static_assert(DNP_UPD_ROWS == 8, "the update's row octet is r0 = 8·(tid / 64)");
static_assert(DNP_F4 % 8 == 0 && DNP_KQ_F4 % 8 == 0, "swizzle groups of eight float4");

static __global__ void __launch_bounds__(DNP_THREADS, 2) delta_net_prefill_state_f32_kernel(
        const float* __restrict__ qk_wave,// Q|K columns of the conv output
        const float* __restrict__ u,      // [h_v, T_tran, D]
        const float* __restrict__ w,      // [h_v, T_tran, D]
        const float* __restrict__ kq,     // [h_v, T_tran, C]
        const float* __restrict__ g_cs,   // [h_v, T_tran]
        float*       __restrict__ o_wave, // [T_wave, h_v·D]
        const long long*    __restrict__ ptrs,
        const unsigned int* __restrict__ spans,
        int n_spans,
        int n_layers_stacked, // > 0: every buffer is layer-stacked
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride, // conv_dim: q and k are strided views of the conv output
        float q_scale) {
    extern __shared__ __align__(16) float smem[];
    float* s_tile = smem;                            // [TV][D]   swizzled
    float* stage  = s_tile + DNP_TV * DNP_DIM;       // [TH][D]   swizzled
    float* skq    = stage + DNP_TH * DNP_DIM;        // [TH][C]   swizzled
    float* vnew   = skq + DNP_TH * DNP_CHUNK;        // [C][TV]
    float* sge    = vnew + DNP_CHUNK * DNP_TV;       // e^{G}
    float* sgd    = sge + DNP_CHUNK;                 // e^{G_last − G}
    __shared__ float s_decay;                        // e^{G_last}
    const float4* s_tile4 = reinterpret_cast<const float4*>(s_tile);
    const float4* stage4  = reinterpret_cast<const float4*>(stage);
    const float4* skq4    = reinterpret_cast<const float4*>(skq);

    // Every operand this pass reads is one the launch carved — the conv output,
    // the scan transients, the output, the span pointers — so a layer stack
    // needs no operand table here, only the slice strides.
    const int layer = (int)blockIdx.z / n_spans;
    const int z = (int)blockIdx.z - layer * n_spans;
    if (n_layers_stacked > 0) {
        const size_t head_rows = (size_t)layer * n_v_heads * t_tran;
        qk_wave += (size_t)layer * t_tran * tok_stride;
        u += head_rows * DNP_DIM;
        w += head_rows * DNP_DIM;
        kq += head_rows * DNP_CHUNK;
        g_cs += head_rows;
        o_wave += head_rows * DNP_DIM;
        ptrs += (size_t)layer * 4 * n_spans;
    }
    const DnSpan sp = dn_span(ptrs, spans, n_spans, z);
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
    const int lane = tid & 31;
    const size_t qk_stride = (size_t)tok_stride;
    const size_t o_stride = (size_t)n_v_heads * DNP_DIM;

    // This head's rows of every operand: span row t is `t` rows past each base.
    const size_t head_row0 = (size_t)h * t_tran + tran;
    const float* __restrict__ w_h  = w + head_row0 * DNP_DIM;
    const float* __restrict__ u_h  = u + head_row0 * DNP_DIM + i_base;
    const float* __restrict__ kq_h = kq + head_row0 * DNP_CHUNK;
    const float* __restrict__ g_h  = g_cs + head_row0;
    const float* __restrict__ q_h  = qk + kh * DNP_DIM;
    const float* __restrict__ k_h  = qk + (n_k_heads + kh) * DNP_DIM;
    float* __restrict__ o_h = o + (size_t)h * DNP_DIM + i_base;

    // The dot phases' 4×2 tile (see dnp_dot42), on warps 0..3: warp → 16
    // tokens × 16 rows, quarter-warp → one token quad tq + {0, 8, 16, 24},
    // lane → rows {ra, ra + 8}. Warps 4..7 sit the dots out: the dots are
    // bound by shared-load wavefronts, not by issue, and four warps cover a
    // half's 32 × 32 outputs.
    const bool dot_warp = warp < DNP_DOT_WARPS;
    const int ra = (warp & 1) * 16 + (lane & 7);
    const int tw = ((warp >> 1) & 1) * 4;  // the warp's first token of the half
    const int tq = tw + (lane >> 3);
    // The update's ownership: columns {ja, ja + 64} of rows r0..r0+7.
    const int ja = tid & 63;
    const int r0 = (tid >> 6) * 8;

    // Load the tile into the swizzled layout. Every load is issued before the
    // first store: the compiler does not hoist a global load past a shared
    // store, so an interleaved loop pays one dependent DRAM round trip per
    // element.
    {
        float4 v[DNP_STAGE_F4];
        #pragma unroll
        for (int k = 0; k < DNP_STAGE_F4; ++k) {
            const int r = warp + 8 * k;
            v[k] = *reinterpret_cast<const float4*>(
                state + ((size_t)h * DNP_DIM + (i_base + r)) * DNP_DIM + 4 * lane);
        }
        #pragma unroll
        for (int k = 0; k < DNP_STAGE_F4; ++k) {
            const int r = warp + 8 * k;
            reinterpret_cast<float4*>(s_tile)[r * DNP_F4 + dnp_swz(r, lane)] = v[k];
        }
    }

    // The operand the NEXT step stages, in flight while this one computes:
    // `pre` holds a half's rows of w, q or k (dnp_fetch_rows), `pkq` its kq
    // rows when it is a q half, and `g_v`/`g_l` the next chunk's G at row
    // `tid` and at its last row.
    float4 pre[DNP_STAGE_F4];
    float4 pkq[DNP_KQ_F4_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < DNP_STAGE_F4; ++k) pre[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    #pragma unroll
    for (int k = 0; k < DNP_KQ_F4_PER_THREAD; ++k) pkq[k] = make_float4(0.f, 0.f, 0.f, 0.f);
    float g_v = 0.f;
    float g_l = 0.f;

    const int n_chunks = (t_len + DNP_CHUNK - 1) / DNP_CHUNK;
    if (n_chunks > 0) {
        const int c0 = min(DNP_CHUNK, t_len);
        dnp_fetch_rows(w_h, DNP_DIM, min(DNP_TH, c0), warp, lane, pre);
        if (tid < c0) g_v = g_h[tid];
        g_l = g_h[c0 - 1];
    }

    float acc3[2 * DNP_UPD_ROWS];
    for (int n = 0; n < n_chunks; ++n) {
        const int t0 = n * DNP_CHUNK;
        const int c_len = min(DNP_CHUNK, t_len - t0);

        // Every step below is: barrier (the previous step's reads of what this
        // one overwrites are done), store the operand that arrived during the
        // previous step, barrier, issue the next step's loads, compute. A
        // phase stages HALF a chunk at a time (DNP_TH tokens), which is what
        // fits two blocks on an SM.

        // ---- phase 1: stage w; v_new = u − w·Sᵀ (block-local) ----
        for (int hb = 0; hb < c_len; hb += DNP_TH) {
            const int hlen = min(DNP_TH, c_len - hb);
            __syncthreads();
            if (hb == 0) {
                if (tid == 0) s_decay = expf(g_l);
                if (tid < c_len) {
                    sge[tid] = expf(g_v);
                    sgd[tid] = expf(g_l - g_v); // G decreases, so the exponent is ≤ 0
                }
            }
            #pragma unroll
            for (int k = 0; k < DNP_STAGE_F4; ++k) dnp_put_row4(stage, warp + 8 * k, lane, pre[k]);
            __syncthreads();
            if (hb + DNP_TH < c_len) {
                dnp_fetch_rows(w_h + (size_t)(t0 + hb + DNP_TH) * DNP_DIM, DNP_DIM,
                               min(DNP_TH, c_len - hb - DNP_TH), warp, lane, pre);
            } else {
                dnp_fetch_rows(q_h + (size_t)t0 * qk_stride, qk_stride, min(DNP_TH, c_len),
                               warp, lane, pre);
                dnp_fetch_kq(kq_h + (size_t)t0 * DNP_CHUNK, min(DNP_TH, c_len), tid, pkq);
            }
            // A warp with no token in this half skips the dot (warp-uniform).
            if (dot_warp && tw < hlen) {
                // u for the eight outputs, ahead of the dot that hides its latency.
                float uv[4][2];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int t = tq + 8 * k;
                    uv[k][0] = 0.f;
                    uv[k][1] = 0.f;
                    if (t < hlen) {
                        const float* urow = u_h + (size_t)(t0 + hb + t) * DNP_DIM;
                        uv[k][0] = urow[ra];
                        uv[k][1] = urow[ra + 8];
                    }
                }
                float acc[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
                dnp_dot42(s_tile4, stage4, ra, tq, acc);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int t = tq + 8 * k;
                    if (t < hlen) {
                        vnew[(hb + t) * DNP_TV + ra] = uv[k][0] - acc[k][0];
                        vnew[(hb + t) * DNP_TV + ra + 8] = uv[k][1] - acc[k][1];
                    }
                }
            }
        }

        // ---- phase 2: stage q (scaled) + kq; o = e^G·(q·Sᵀ) + Σ_{s≤t} kq·v_new ----
        for (int hb = 0; hb < c_len; hb += DNP_TH) {
            const int hlen = min(DNP_TH, c_len - hb);
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < DNP_STAGE_F4; ++k) {
                float4 v = pre[k];
                v.x = v.x * q_scale;
                v.y = v.y * q_scale;
                v.z = v.z * q_scale;
                v.w = v.w * q_scale;
                dnp_put_row4(stage, warp + 8 * k, lane, v);
            }
            #pragma unroll
            for (int k = 0; k < DNP_KQ_F4_PER_THREAD; ++k) {
                const int e = tid + k * DNP_THREADS;
                const int t = e / DNP_KQ_F4;
                reinterpret_cast<float4*>(skq)[t * DNP_KQ_F4 + dnp_swz(t, e % DNP_KQ_F4)] = pkq[k];
            }
            __syncthreads();
            if (hb + DNP_TH < c_len) {
                const int t1 = t0 + hb + DNP_TH;
                const int n1 = min(DNP_TH, c_len - hb - DNP_TH);
                dnp_fetch_rows(q_h + (size_t)t1 * qk_stride, qk_stride, n1, warp, lane, pre);
                dnp_fetch_kq(kq_h + (size_t)t1 * DNP_CHUNK, n1, tid, pkq);
            } else {
                dnp_fetch_rows(k_h + (size_t)t0 * qk_stride, qk_stride, min(DNP_TH, c_len),
                               warp, lane, pre);
            }
            if (dot_warp && tw < hlen) { // as phase 1: a warp with no token skips the dots
                float acc[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
                dnp_dot42(s_tile4, stage4, ra, tq, acc); // pre-update S
                // The intra-chunk read of this chunk's own writes: Σ_{s ≤ tc}
                // kq[tc][s]·v_new[s][r], s ascending, four kq columns per load
                // (one quarter-warp-broadcast load per token, swizzle tq & 7).
                const int s_last = hb + min(tw + 24 + 3, hlen - 1); // the warp's longest sum
                const int zt = tq & 7;
                float in[4][2] = {{0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}};
                #pragma unroll
                for (int m = 0; m < DNP_KQ_F4 / 8; ++m) {
                    #pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        const int s4 = 8 * m + c;
                        if (4 * s4 <= s_last) {
                            float4 kk[4];
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                kk[k] = skq4[(tq + 8 * k) * DNP_KQ_F4 + 8 * m + (c ^ zt)];
                            }
                            #pragma unroll
                            for (int e = 0; e < 4; ++e) {
                                const int s = 4 * s4 + e;
                                const float va = vnew[s * DNP_TV + ra];
                                const float vb = vnew[s * DNP_TV + ra + 8];
                                #pragma unroll
                                for (int k = 0; k < 4; ++k) {
                                    if (s <= hb + tq + 8 * k) {
                                        const float ke = dnp_f4_at(kk[k], e);
                                        in[k][0] += ke * va;
                                        in[k][1] += ke * vb;
                                    }
                                }
                            }
                        }
                    }
                }
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const int t = tq + 8 * k;
                    if (t < hlen) {
                        const int tc = hb + t; // within-chunk token index
                        float* orow = o_h + (size_t)(t0 + tc) * o_stride;
                        orow[ra] = acc[k][0] * sge[tc] + in[k][0];
                        orow[ra + 8] = acc[k][1] * sge[tc] + in[k][1];
                    }
                }
            }
        }

        // ---- phase 3: stage k ⊙ e^{G_last−G}; S ← e^{G_last}·S + v_newᵀ(k ⊙ e^{G_last−G}) ----
        // Each thread owns 16 S elements — columns {ja, ja + 64} of rows
        // r0..r0+7 — disjoint across threads, so the in-place update has no
        // races. The accumulators persist across the staged halves; per token,
        // two stage loads and two warp-broadcast v_new loads feed 16 FMAs.
        #pragma unroll
        for (int rr = 0; rr < 2 * DNP_UPD_ROWS; ++rr) acc3[rr] = 0.f;
        for (int hb = 0; hb < c_len; hb += DNP_TH) {
            const int hlen = min(DNP_TH, c_len - hb);
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < DNP_STAGE_F4; ++k) {
                const int t = warp + 8 * k;
                const float e = sgd[hb + t];
                float4 v = pre[k];
                v.x = v.x * e;
                v.y = v.y * e;
                v.z = v.z * e;
                v.w = v.w * e;
                dnp_put_row4(stage, t, lane, v);
            }
            __syncthreads();
            if (hb + DNP_TH < c_len) {
                dnp_fetch_rows(k_h + (size_t)(t0 + hb + DNP_TH) * qk_stride, qk_stride,
                               min(DNP_TH, c_len - hb - DNP_TH), warp, lane, pre);
            } else if (n + 1 < n_chunks) {
                const int t1 = t0 + DNP_CHUNK;
                const int c1 = min(DNP_CHUNK, t_len - t1);
                dnp_fetch_rows(w_h + (size_t)t1 * DNP_DIM, DNP_DIM, min(DNP_TH, c1), warp, lane,
                               pre);
                if (tid < c1) g_v = g_h[t1 + tid];
                g_l = g_h[t1 + c1 - 1];
            }
            #pragma unroll
            for (int m = 0; m < DNP_TH / 8; ++m) {
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int t = 8 * m + c;
                    if (t < hlen) {
                        // Column ja of swizzled row t (t & 7 == c); ja + 64 is
                        // 16 float4 slots on, past the swizzle's reach.
                        const float* krow =
                            stage + t * DNP_DIM + 4 * ((ja >> 2) ^ c) + (ja & 3);
                        const float kga = krow[0];
                        const float kgb = krow[64];
                        const float4* vrow =
                            reinterpret_cast<const float4*>(&vnew[(hb + t) * DNP_TV + r0]);
                        const float4 v0 = vrow[0];
                        const float4 v1 = vrow[1];
                        acc3[0] += v0.x * kga;
                        acc3[1] += v0.y * kga;
                        acc3[2] += v0.z * kga;
                        acc3[3] += v0.w * kga;
                        acc3[4] += v1.x * kga;
                        acc3[5] += v1.y * kga;
                        acc3[6] += v1.z * kga;
                        acc3[7] += v1.w * kga;
                        acc3[8] += v0.x * kgb;
                        acc3[9] += v0.y * kgb;
                        acc3[10] += v0.z * kgb;
                        acc3[11] += v0.w * kgb;
                        acc3[12] += v1.x * kgb;
                        acc3[13] += v1.y * kgb;
                        acc3[14] += v1.z * kgb;
                        acc3[15] += v1.w * kgb;
                    }
                }
            }
        }
        {
            const float dec = s_decay;
            #pragma unroll
            for (int rr = 0; rr < DNP_UPD_ROWS; ++rr) {
                // Row r0 + rr: its swizzle is rr, r0 being a multiple of 8.
                float* sa = &s_tile[(r0 + rr) * DNP_DIM + 4 * ((ja >> 2) ^ rr) + (ja & 3)];
                sa[0] = sa[0] * dec + acc3[rr];
                sa[64] = sa[64] * dec + acc3[DNP_UPD_ROWS + rr];
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
    {
        // Every shared read first, then the stores — as the load above.
        float4 v[DNP_STAGE_F4];
        #pragma unroll
        for (int k = 0; k < DNP_STAGE_F4; ++k) {
            const int r = warp + 8 * k;
            v[k] = s_tile4[r * DNP_F4 + dnp_swz(r, lane)];
        }
        #pragma unroll
        for (int k = 0; k < DNP_STAGE_F4; ++k) {
            const int r = warp + 8 * k;
            *reinterpret_cast<float4*>(
                state_out + ((size_t)h * DNP_DIM + (i_base + r)) * DNP_DIM + 4 * lane) = v[k];
        }
    }
}

// 0 when launched, 1 when the shape is refused (nothing written): the window
// past DNC_KMAX among the refusals, which the host's own bound
// (`DELTA_NET_PREFILL_CONV_MAX`) must agree with — a refusal it does not
// expect is an error there, never an unwritten output.
static inline int launch_conv_prefill_f32(
        const float* x_wave,
        const float* kernel,
        float* y_wave,
        const long long* ptrs,
        const unsigned int* spans,
        int n_spans,
        const DnLayerOps* layers,
        int n_layers,
        int t_wave,
        int max_len,
        int channels,
        int kwidth,
        int qk_channels,
        float eps,
        cudaStream_t stream) {
    if (n_spans <= 0 || max_len <= 0 || channels <= 0 || kwidth <= 1 || kwidth > DNC_KMAX) return 1;
    // A layer stack needs its operand table; a single layer reads the args.
    if (n_layers <= 0 || (n_layers > 1 && layers == nullptr)) return 1;
    if ((long long)n_layers * n_spans > 65535) return 1;
    // The epilogue's norm reduction is block-local; a block must hold whole
    // head groups, which qk_channels = h_k·256 guarantees at 256 threads.
    if (qk_channels < 0 || qk_channels > channels || qk_channels % 256 != 0) return 1;
    const int threads = 256;
    dim3 grid((channels + threads - 1) / threads, (max_len + DNC_TOK - 1) / DNC_TOK,
              n_layers * n_spans);
    delta_net_conv_prefill_f32_kernel<<<grid, threads, 0, stream>>>(
        x_wave, kernel, y_wave, ptrs, spans, n_spans, layers, t_wave, channels,
        kwidth, qk_channels, eps);
    return 0;
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
        const DnLayerOps* layers,
        int n_layers,
        int max_len,
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        cudaStream_t stream) {
    if (n_spans <= 0 || max_len <= 0 || n_v_heads <= 0 || n_k_heads <= 0) return;
    if (n_layers <= 0 || (n_layers > 1 && layers == nullptr)) return;
    if ((long long)n_layers * n_spans > 65535) return;
    // The row buffers hold the longest chunk any block of this launch walks,
    // rounded up to the A/kq grid's 4-wide j-tile: a tile reads k rows jt..jt+3
    // whole, past c_len when c_len is not a multiple of 4, and those reads must
    // land inside the buffer (their products are discarded, the loads are not).
    const int rows = max_len < DNP_CHUNK ? ((max_len + 3) & ~3) : DNP_CHUNK;
    const auto smem_for = [](int r) {
        return (2 * r * DNP_DIM + r * DNI_ALD + 2 * DNP_CHUNK) * (int)sizeof(float);
    };
    const int smem_bytes = smem_for(rows);
    // A full chunk's 81.5 KiB exceeds the 48 KiB default dynamic-smem ceiling; the
    // ceiling is raised once to the full-chunk size, which covers every launch.
    // The attribute is per-function and sticky, so a redundant set is a no-op.
    static int smem_raised = 0;
    if (!smem_raised) {
        cudaFuncSetAttribute(delta_net_prefill_intra_f32_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_for(DNP_CHUNK));
        smem_raised = 1;
    }
    // Chunks for the WIDEST span: shorter spans' surplus blocks return at the
    // top of the kernel. A rectangle wastes at most `max_len − len` block rows
    // per span, which is nothing against the launch it replaces.
    const int n_chunks = (max_len + DNP_CHUNK - 1) / DNP_CHUNK;
    dim3 grid(n_chunks, n_v_heads, n_layers * n_spans);
    delta_net_prefill_intra_f32_kernel<<<grid, DNP_THREADS, smem_bytes, stream>>>(
        qk_wave, v_wave, alpha_wave, blin_wave, dt_bias, a_neg, u, w, kq, g_cs,
        spans, n_spans, layers, t_tran, n_v_heads, n_k_heads, tok_stride,
        q_scale, rows);
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
        const DnLayerOps* layers, // read for presence only: see the kernel
        int n_layers,
        int t_tran,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        cudaStream_t stream) {
    if (n_spans <= 0 || n_v_heads <= 0 || n_k_heads <= 0) return;
    const bool stacked = layers != nullptr;
    if (n_layers <= 0 || (n_layers > 1 && !stacked)) return;
    if ((long long)n_layers * n_spans > 65535) return;
    // 48.5 KB: two blocks share an SM's 100 KB (less 1 KB of driver reserve
    // each). It is past the 48 KB default dynamic-smem ceiling, so the
    // ceiling is raised once, along with the carveout preference that keeps
    // the whole 100 KB as shared memory; both attributes are per-function and
    // sticky.
    const int smem_bytes = DNP_STATE_SMEM_FLOATS * (int)sizeof(float);
    static int attrs_set = 0;
    if (!attrs_set) {
        cudaFuncSetAttribute(delta_net_prefill_state_f32_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        cudaFuncSetAttribute(delta_net_prefill_state_f32_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        attrs_set = 1;
    }
    // No span dimension in the grid's extent beyond `n_spans`: this pass walks
    // its span's chunks serially inside the block, so its shape never depended
    // on the length.
    dim3 grid(n_v_heads, DNP_DIM / DNP_TV, n_layers * n_spans);
    delta_net_prefill_state_f32_kernel<<<grid, 256, smem_bytes, stream>>>(
        qk_wave, u, w, kq, g_cs, o_wave, ptrs, spans, n_spans,
        stacked ? n_layers : 0, t_tran, n_v_heads, n_k_heads, tok_stride,
        q_scale);
}

} // namespace delta_net
