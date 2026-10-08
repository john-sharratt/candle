#pragma once
// Gated DeltaNet — the short-span scan: the causal conv, the intra-chunk solve
// and the state pass of the prefill scan in ONE launch, for spans of at most
// DNS_ROWS rows (a speculative verify block: the drafts plus the accepted
// token).
//
// At that width the three-kernel path (delta_net_prefill_kernel.cuh) is three
// tiny latency-bound grids chained through global memory — conv output, then
// u/w/kq/G, then the state pass — each paying a launch and a dependent DRAM
// round trip for a few rows of work. Here one block per (V head, d_v tile,
// span) — the state pass's grid — does all of it with the operands in shared
// memory:
//
//   (a) the S tile load is issued first, before any other load or store, so
//       its DRAM latency hides behind everything up to the state phases;
//   (b) the conv rows the block needs: K head kh = h % h_k's q and k channels
//       (SiLU + the 128-wide l2-norm tree), and the tile's 32 v channels
//       (SiLU) — recomputed by every block that reads them rather than
//       handed through a conv-output buffer;
//   (c) gates, the G scan, the A / kq grid and the forward-substitution solve
//       for w (all 128 columns) and u (the tile's 32);
//   (d) v_new = u − w·Sᵀ, o = e^G·(q·Sᵀ) + Σ kq·v_new, and the S update —
//       the state pass's three phases over a single chunk;
//   (e) the advanced tile and the `o` rows.
//
// **Bit-identical to the three-kernel path, by construction.** Every value is
// the same operation sequence on the same operands: the conv's j-ascending FMA
// chain, the epilogue through the same helpers (delta_net_common.cuh) and the
// same reduction tree, the same Hillis–Steele scan over the chunk's 64 slots,
// one sequential d-ascending FMA chain per A/kq entry, the same substitution
// order, and the state pass's phases through `dnp_dot4` and its own
// accumulation order. Only where a value lives (registers and shared memory
// instead of global buffers) and which thread computes it differ — neither
// changes a bit. `a_short_span_fused_launch_is_the_three_launches_bit_for_bit`
// (delta_net/cuda.rs) holds it.
//
// What it writes: the span's `o` rows, every element of the advanced state
// (`state_out`, which may be the entering buffer — each block reads its tile
// before it stores), and the advanced conv tail (`tail_out`, a separate buffer
// from the entering tail every block reads), each element by exactly one
// block. It writes no conv-output rows: this kernel is the only consumer of
// the span's conv output, so it is never materialised.
//
// Concrete (non-template) and `static`, as the prefill header: compiled by the
// single translation unit delta_net_api_f32.cu.

#include "delta_net_prefill_kernel.cuh"

// The longest span the fused kernel takes — every per-row buffer below is
// sized by it. Must match `DELTA_NET_SHORT_SPAN_ROWS` in api.rs.
#define DNS_ROWS 8
// The causal conv width the kernel is compiled for. Must match
// `DELTA_NET_SHORT_SPAN_CONV` in api.rs.
#define DNS_KW 4
#define DNS_TAIL (DNS_KW - 1)
#define DNS_THREADS 256
// Conv input rows a span reads: its own plus the tail's.
#define DNS_XROWS (DNS_ROWS + DNS_TAIL)

// The thread maps below: one v (row, column) per thread, the A/kq grid on the
// block's last DNS_ROWS² threads clear of the 160 solve columns, and the norm
// scratch inside the S tile it precedes.
static_assert(DNS_THREADS == DNS_ROWS * DNP_TV, "one v (row, column) per thread");
static_assert(DNS_THREADS == 2 * DNP_DIM, "one q or k channel per thread");
static_assert(DNP_DIM + DNP_TV <= DNS_THREADS - DNS_ROWS * DNS_ROWS,
              "the grid threads overlap the solve columns");
static_assert(DNS_ROWS * DNS_THREADS <= DNP_TV * DNP_SLD,
              "the norm scratch does not fit the S tile");
static_assert(DNS_ROWS <= 32, "the G scan runs in one warp");

namespace delta_net {

static __global__ void __launch_bounds__(DNS_THREADS, 2) delta_net_short_span_f32_kernel(
        const float* __restrict__ x_wave,     // [T_wave, C] raw conv input (QKV)
        const float* __restrict__ kernel,     // [C, DNS_KW]
        const float* __restrict__ alpha_wave, // [T_wave, h_v] raw
        const float* __restrict__ blin_wave,  // [T_wave, h_v] raw
        const float* __restrict__ dt_bias,    // [h_v]
        const float* __restrict__ a_neg,      // [h_v]
        float*       __restrict__ o_wave,     // [T_wave, h_v·D]
        const long long*    __restrict__ ptrs,  // [4, n_spans]
        const unsigned int* __restrict__ spans, // [2, n_spans]
        int n_spans,
        int n_v_heads,
        int n_k_heads,
        int channels,   // conv_dim = (2·h_k + h_v)·D
        float eps,
        float q_scale) {
    // The S tile, in the state pass's padded layout. Before the tile is stored
    // it is the norm reduction's scratch: [DNS_ROWS][DNS_THREADS] sums of
    // squares, one row of the block's 256 Q|K channels per token.
    __shared__ __align__(16) float s_tile[DNP_TV * DNP_SLD];
    __shared__ __align__(16) float sq[DNS_ROWS * DNP_SLD]; // q · q_scale
    __shared__ __align__(16) float sk[DNS_ROWS * DNP_SLD]; // k
    __shared__ __align__(16) float sw[DNS_ROWS * DNP_SLD]; // w (solve, all D columns)
    __shared__ __align__(16) float vnew[DNS_ROWS * DNP_VLD];
    __shared__ float sv[DNS_ROWS * DNP_TV]; // the tile's v columns, post-SiLU
    __shared__ float su[DNS_ROWS * DNP_TV]; // u (solve, the tile's columns)
    __shared__ float sA[DNS_ROWS * DNS_ROWS];
    __shared__ float skq[DNS_ROWS * DNS_ROWS];
    __shared__ float sg[DNS_ROWS];  // G, the within-chunk log-decay cumsum
    __shared__ float sb[DNS_ROWS];  // β
    __shared__ float sge[DNS_ROWS]; // e^{G}
    __shared__ float sgd[DNS_ROWS]; // e^{G_last − G}
    __shared__ float snorm[DNS_ROWS * 2]; // Σx² per token, q head then k head
    __shared__ float s_decay;            // e^{G_last}
    float* red = s_tile;

    const DnSpan sp = dn_span(ptrs, spans, n_spans, (int)blockIdx.z);
    const int len = sp.len;
    // The launcher admits the table by its longest span and every row buffer
    // is sized by DNS_ROWS; a span outside 1..=DNS_ROWS is a table that
    // disagrees with the extent it was launched under, and stops the kernel
    // rather than overrunning those buffers or leaving `state_out` unwritten.
    if (len < 1 || len > DNS_ROWS) __trap();

    const int h = blockIdx.x;
    const int kh = h % n_k_heads; // ggml's tiled GQA broadcast
    const int i_base = (int)blockIdx.y * DNP_TV;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int qk_channels = 2 * n_k_heads * DNP_DIM;
    const float* __restrict__ x = x_wave + (size_t)sp.start * channels;
    const float* __restrict__ tail = sp.tail;
    const size_t o_stride = (size_t)n_v_heads * DNP_DIM;
    float* __restrict__ o = o_wave + (size_t)sp.start * o_stride;

    // ---- (a) the S tile, first: thread → column j of rows r0..r0+15 -------
    // The same (row, column) ownership as the state update below, so the
    // entering values stay in registers for it; lanes take consecutive j, so
    // each row load is coalesced.
    const int j = tid & (DNP_DIM - 1);
    const int r0 = (tid >> 7) * (DNP_TV / 2);
    float s_reg[DNP_TV / 2];
    #pragma unroll
    for (int rr = 0; rr < DNP_TV / 2; ++rr) {
        s_reg[rr] = sp.state[((size_t)h * DNP_DIM + (i_base + r0 + rr)) * DNP_DIM + j];
    }

    // ---- every other global load, before any store -----------------------
    // Q|K: thread → one channel of K head kh — q for the first 128 threads,
    // k for the rest, lane d of the head — over every row of the span.
    const int qk_ch = tid < DNP_DIM ? kh * DNP_DIM + tid
                                    : (n_k_heads + kh) * DNP_DIM + (tid - DNP_DIM);
    float qk_w[DNS_KW];
    #pragma unroll
    for (int jj = 0; jj < DNS_KW; ++jj) qk_w[jj] = kernel[(size_t)qk_ch * DNS_KW + jj];
    // in(p − DNS_TAIL): x for p ≥ DNS_TAIL, the entering tail before it.
    float qk_x[DNS_XROWS];
    #pragma unroll
    for (int p = 0; p < DNS_XROWS; ++p) {
        const int idx = p - DNS_TAIL;
        qk_x[p] = 0.f;
        if (p < len + DNS_TAIL) {
            qk_x[p] = (idx >= 0) ? x[(size_t)idx * channels + qk_ch]
                                 : tail[(size_t)qk_ch * DNS_TAIL + (DNS_TAIL + idx)];
        }
    }
    // V: thread → (row v_t, tile column v_r) — 32 columns × 8 rows.
    const int v_r = tid & (DNP_TV - 1);
    const int v_t = tid / DNP_TV;
    const int v_ch = qk_channels + h * DNP_DIM + i_base + v_r;
    float v_w[DNS_KW];
    float v_x[DNS_KW];
    #pragma unroll
    for (int jj = 0; jj < DNS_KW; ++jj) {
        v_w[jj] = 0.f;
        v_x[jj] = 0.f;
        if (v_t < len) {
            const int idx = v_t - DNS_TAIL + jj;
            v_w[jj] = kernel[(size_t)v_ch * DNS_KW + jj];
            v_x[jj] = (idx >= 0) ? x[(size_t)idx * channels + v_ch]
                                 : tail[(size_t)v_ch * DNS_TAIL + (DNS_TAIL + idx)];
        }
    }
    // Gates: warp 0, lane → row.
    float g_alpha = 0.f, g_blin = 0.f, g_dt = 0.f, g_a = 0.f;
    if (warp == 0 && lane < len) {
        const size_t row = (size_t)(sp.start + lane) * n_v_heads + h;
        g_alpha = alpha_wave[row];
        g_blin = blin_wave[row];
        g_dt = dt_bias[h];
        g_a = a_neg[h];
    }

    // ---- the advanced conv tail -------------------------------------------
    // The RAW inputs the next window starts from. Every block of the span
    // shares the copy, one element each, so each element has one writer.
    {
        const int nblk = (int)(gridDim.x * gridDim.y);
        const int b = (int)(blockIdx.y * gridDim.x + blockIdx.x);
        const int total = channels * DNS_TAIL;
        for (int e = b * DNS_THREADS + tid; e < total; e += nblk * DNS_THREADS) {
            const int c = e / DNS_TAIL;
            const int jj = e - c * DNS_TAIL;
            const int idx = len - DNS_TAIL + jj;
            sp.tail_out[e] = (idx >= 0) ? x[(size_t)idx * channels + c]
                                        : tail[(size_t)c * DNS_TAIL + (DNS_TAIL + idx)];
        }
    }

    // ---- (b) the conv rows ------------------------------------------------
    // y[t] = epilogue( Σ_j w[j] · in(t − (K−1) + j) ), j ascending — the conv
    // kernel's chain.
    float qk_sv[DNS_ROWS];
    #pragma unroll
    for (int t = 0; t < DNS_ROWS; ++t) {
        qk_sv[t] = 0.f;
        if (t < len) {
            float acc = 0.f;
            #pragma unroll
            for (int jj = 0; jj < DNS_KW; ++jj) acc += qk_w[jj] * qk_x[t + jj];
            const float s = dn_silu(acc);
            qk_sv[t] = s;
            red[t * DNS_THREADS + tid] = s * s;
        }
    }
    if (v_t < len) {
        float acc = 0.f;
        #pragma unroll
        for (int jj = 0; jj < DNS_KW; ++jj) acc += v_w[jj] * v_x[jj];
        sv[v_t * DNP_TV + v_r] = dn_silu(acc);
    }
    // Gates and the inclusive scan over the chunk's 64 slots, as the intra
    // pass runs it: slot t adds slot t − off at each doubling, rows past the
    // span scan zeros. Slots past 31 only ever feed slots past 31, so warp 0's
    // 32 lanes carry every slot a row reads; at off = 32 every lane adds the
    // zero the intra pass adds there.
    if (warp == 0) {
        float g = 0.f;
        float bv = 0.f;
        if (lane < len) {
            g = dn_decay_gate(g_a, g_alpha, g_dt);
            bv = dn_sigmoid(g_blin);
        }
        for (int off = 1; off < DNP_CHUNK; off <<= 1) {
            const float up = __shfl_up_sync(0xffffffffu, g, off & 31);
            float add = 0.f;
            if (lane >= off) add = up;
            g += add;
        }
        if (lane < DNS_ROWS) {
            sg[lane] = g;
            sb[lane] = bv;
        }
    }
    __syncthreads();

    // The l2-norm tree of dn_silu_norm_epilogue, every row at once: halve the
    // 128-wide group through shared memory twice, then the last five levels
    // within the group's first warp — slot l += slot l + off for l < off at
    // every level, so the root is the same sum.
    if ((tid & (DNP_DIM - 1)) < DNP_DIM / 2) {
        #pragma unroll
        for (int t = 0; t < DNS_ROWS; ++t) {
            if (t < len) red[t * DNS_THREADS + tid] += red[t * DNS_THREADS + tid + DNP_DIM / 2];
        }
    }
    __syncthreads();
    if ((tid & (DNP_DIM - 1)) < 32) {
        #pragma unroll
        for (int t = 0; t < DNS_ROWS; ++t) {
            if (t < len) {
                float v = red[t * DNS_THREADS + tid] + red[t * DNS_THREADS + tid + 32];
                #pragma unroll
                for (int off = 16; off >= 1; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) snorm[t * 2 + (tid >> 7)] = v;
            }
        }
    }
    __syncthreads();

    // The normed rows (q scaled on store, as the scan passes read it), the
    // chunk's decay vectors, and the S tile — over the reduction scratch,
    // whose last read was before the barrier.
    #pragma unroll
    for (int t = 0; t < DNS_ROWS; ++t) {
        if (t < len) {
            const float y = dn_l2_scale(qk_sv[t], snorm[t * 2 + (tid >> 7)], eps);
            if (tid < DNP_DIM) sq[t * DNP_SLD + tid] = y * q_scale;
            else               sk[t * DNP_SLD + (tid - DNP_DIM)] = y;
        }
    }
    if (tid < len) {
        const float gv = sg[tid];
        const float gl = sg[len - 1];
        sge[tid] = expf(gv);
        sgd[tid] = expf(gl - gv); // G decreases, so the exponent is ≤ 0
    }
    if (tid == 0) s_decay = expf(sg[len - 1]);
    #pragma unroll
    for (int rr = 0; rr < DNP_TV / 2; ++rr) s_tile[(r0 + rr) * DNP_SLD + j] = s_reg[rr];
    __syncthreads();

    // ---- (c) A / kq grid and the right-hand sides -------------------------
    // The last two warps take the grid, one (i, j ≤ i) entry per thread, each
    // a single d-ascending FMA chain — the per-accumulator order of the intra
    // pass's 4-wide tiles. The first 160 threads build the right-hand sides
    // meanwhile: columns 0..127 for w (βk ⊙ e^G), 128..159 for u (βv) over the
    // tile's v columns.
    if (tid >= DNS_THREADS - DNS_ROWS * DNS_ROWS) {
        const int p = tid - (DNS_THREADS - DNS_ROWS * DNS_ROWS);
        const int i = p / DNS_ROWS;
        const int jc = p % DNS_ROWS;
        if (jc <= i && i < len) {
            const float* ki = &sk[i * DNP_SLD];
            const float* qi = &sq[i * DNP_SLD];
            const float* kj = &sk[jc * DNP_SLD];
            float dkk = 0.f;
            float dqk = 0.f;
            #pragma unroll 8
            for (int d = 0; d < DNP_DIM; ++d) {
                const float kiv = ki[d];
                const float qiv = qi[d];
                const float kjv = kj[d];
                dkk += kiv * kjv;
                dqk += qiv * kjv;
            }
            const float dec = expf(fminf(0.f, sg[i] - sg[jc]));
            if (jc < i) sA[i * DNS_ROWS + jc] = sb[i] * dkk * dec;
            skq[i * DNS_ROWS + jc] = dqk * dec;
        }
    }
    const bool solver = tid < DNP_DIM + DNP_TV;
    const bool is_w = tid < DNP_DIM;
    float xr[DNS_ROWS];
    #pragma unroll
    for (int i = 0; i < DNS_ROWS; ++i) {
        float b = 0.f;
        if (solver && i < len) {
            b = is_w ? sb[i] * sk[i * DNP_SLD + tid] * expf(sg[i])
                     : sb[i] * sv[i * DNP_TV + (tid - DNP_DIM)];
        }
        xr[i] = b;
    }
    __syncthreads(); // A complete before the substitution reads it

    // (I + A) x = b  →  x[i] = b[i] − Σ_{j<i} A[i][j] x[j], the intra pass's
    // order, stopping at the span's length.
    if (solver) {
        #pragma unroll
        for (int i = 1; i < DNS_ROWS; ++i) {
            if (i >= len) break;
            float acc = xr[i];
            #pragma unroll
            for (int jc = 0; jc < i; ++jc) acc -= sA[i * DNS_ROWS + jc] * xr[jc];
            xr[i] = acc;
        }
        #pragma unroll
        for (int i = 0; i < DNS_ROWS; ++i) {
            if (i < len) {
                if (is_w) sw[i * DNP_SLD + tid] = xr[i];
                else      su[i * DNP_TV + (tid - DNP_DIM)] = xr[i];
            }
        }
    }
    __syncthreads();

    // ---- (d) the state pass over the one chunk ----------------------------
    // Phase 1 and phase 2's inter-chunk read share a pass: both read the
    // pre-update S, and neither reads the other. warp → t, lane → r, as the
    // state kernel maps them.
    float inter = 0.f;
    if (warp < len) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        dnp_dot4(&s_tile[lane * DNP_SLD], sw, warp, len, acc);
        float qs[4] = {0.f, 0.f, 0.f, 0.f};
        dnp_dot4(&s_tile[lane * DNP_SLD], sq, warp, len, qs);
        vnew[warp * DNP_VLD + lane] = su[warp * DNP_TV + lane] - acc[0];
        inter = qs[0];
    }
    __syncthreads(); // v_new complete

    // Phase 2's intra-chunk read and the output row.
    if (warp < len) {
        const int tc = warp;
        float intra = 0.f;
        for (int s = 0; s <= tc; ++s) {
            intra += skq[tc * DNS_ROWS + s] * vnew[s * DNP_VLD + lane];
        }
        o[(size_t)tc * o_stride + (size_t)h * DNP_DIM + (i_base + lane)] =
            inter * sge[tc] + intra;
    }

    // Phase 3: S ← e^{G_last}·S + v_newᵀ(k ⊙ e^{G_last−G}), straight to
    // `state_out` from the registers that loaded the entering tile.
    {
        float acc[DNP_TV / 2];
        #pragma unroll
        for (int rr = 0; rr < DNP_TV / 2; ++rr) acc[rr] = 0.f;
        for (int t = 0; t < len; ++t) {
            const float kg = sk[t * DNP_SLD + j] * sgd[t];
            const float4* vrow = reinterpret_cast<const float4*>(&vnew[t * DNP_VLD + r0]);
            #pragma unroll
            for (int r4 = 0; r4 < DNP_TV / 8; ++r4) {
                const float4 vv = vrow[r4];
                acc[4 * r4 + 0] += vv.x * kg;
                acc[4 * r4 + 1] += vv.y * kg;
                acc[4 * r4 + 2] += vv.z * kg;
                acc[4 * r4 + 3] += vv.w * kg;
            }
        }
        const float dec = s_decay;
        #pragma unroll
        for (int rr = 0; rr < DNP_TV / 2; ++rr) {
            sp.state_out[((size_t)h * DNP_DIM + (i_base + r0 + rr)) * DNP_DIM + j] =
                s_reg[rr] * dec + acc[rr];
        }
    }
}

// 0 when launched, 1 when the shape is refused (nothing written): a span
// longer than DNS_ROWS, a conv width other than DNS_KW, or a geometry the
// kernel's channel mapping does not cover.
static inline int launch_short_span_f32(
        const float* x_wave,
        const float* kernel,
        const float* alpha_wave,
        const float* blin_wave,
        const float* dt_bias,
        const float* a_neg,
        float* o_wave,
        const long long* ptrs,
        const unsigned int* spans,
        int n_spans,
        int max_len,
        int n_v_heads,
        int n_k_heads,
        int channels,
        int kwidth,
        float eps,
        float q_scale,
        cudaStream_t stream) {
    if (n_spans <= 0 || n_spans > 65535) return 1;
    if (max_len < 1 || max_len > DNS_ROWS) return 1;
    if (kwidth != DNS_KW) return 1;
    if (n_v_heads <= 0 || n_k_heads <= 0 || n_v_heads % n_k_heads != 0) return 1;
    if (channels != (2 * n_k_heads + n_v_heads) * DNP_DIM) return 1;
    dim3 grid(n_v_heads, DNP_DIM / DNP_TV, n_spans);
    delta_net_short_span_f32_kernel<<<grid, DNS_THREADS, 0, stream>>>(
        x_wave, kernel, alpha_wave, blin_wave, dt_bias, a_neg, o_wave, ptrs, spans,
        n_spans, n_v_heads, n_k_heads, channels, eps, q_scale);
    return 0;
}

} // namespace delta_net
