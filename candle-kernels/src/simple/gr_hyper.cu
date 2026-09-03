// =============================================================================
// GR (Gated Residual) FUSED PRE-MIX / COMBINE
// =============================================================================
// Qwen3.8-Flash-Next carries a **4-stream** residual and has no layer norms:
// every attention and every MoE sub-block is bracketed by a hyper-connection
// that reads the four streams down to one narrow block input and scatters the
// block's output back across them (`docs/qwen38_flash_next.md` §12.3). At 48
// layers × 2 sub-blocks that bracket runs 96 times per forward, and expressed
// as eager tensor ops it was ~13 launches and ~6 full passes over a
// `[rows, 4·2560]` F32 buffer each time — 84 MB a pass at prefill width.
//
// These kernels are the three parts of that bracket which are elementwise or a
// reduction. The two low-rank projections (10240↔320) stay in cuBLAS: they are
// real GEMMs, and §0.4 rule 1 says the cheapest kernel is the one already
// written.
//
//   gr_norm     x[n,hc,d]                     → xn[n,hc*d]
//               grouped RMS (per stream, over d) × the [hc*d] gain, one pass
//   gr_mix      xn[n,hc*d], gate_raw[n,hc*d]  → mixed[n,d]
//               mean over streams of xn ⊙ sigmoid(gate_raw), one pass
//   gr_combine  res[n,hc,d] += out[n,d] · 2·sigmoid(inject[n,hc]/hc)
//               in place, one pass
//
// WHY THESE ARE NEW KERNELS AND NOT AN EXTENSION OF `hyper_mhc.cu`
// ----------------------------------------------------------------
// `hyper_mhc.cu` fuses DeepSeek-V4's mHC, which is a *different algebra*: a
// Sinkhorn-normalised combine matrix and a weighted residual reduction. This
// one is a low-rank sigmoid read gate and a `2·sigmoid` scatter weight centred
// on 1, with no Sinkhorn and no combine matrix (the design doc's §0.9 lift
// table says exactly this: mirror mHC's *discipline*, not its arithmetic).
// Extending mHC would mean a runtime branch through its inner loop for a body
// that shares no arithmetic — which §0.4 rule 2 forbids, because it puts our
// predicate in another model's hot path.
//
// WHY A DENSE BASE POINTER AND NOT A DESCRIPTOR TABLE
// ---------------------------------------------------
// Invariant 2b exists for kernels whose per-row data lives in scattered places,
// where demanding one base pointer forces the caller to `cat` rows together.
// Nothing here is scattered: the wide residual is a single dense
// `[rows, hc, d]` buffer that the wave already owns, and every row of every
// operand is at a computable offset inside it. A descriptor table would be
// machinery for a problem this operation does not have (§0.8).
//
// The Rust `cpu_*` scalar paths in `qwen4exp/hyper.rs` are the reference these
// are asserted against, in the same order, with the same eps placement —
// `hyper_mhc.cu`'s discipline.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

// The SHARED sigmoid, not a local one. `candle_nn::ops::sigmoid` — the eager
// path these kernels replace — routes to `fast_exp::sigmoid<float>`, a cubic
// polynomial with ~0.009% error, and a fusion's job is to be the same
// computation faster rather than a different one.
//
// This mattered in practice. The first cut here rolled its own
// `1/(1 + __expf(-x))`, which is ~2 ulp — about 400× MORE accurate than the
// reference. That is still a ~9e-5 relative change on every gate value, which
// is orders of magnitude above any reassociation effect, and it moved a
// marginal session across the KV calibration's C10 edge. The cost was paid in
// compression ratio at C5, the level production runs, to fix a rung that
// exists only as a probe. Improving precision inside a performance change also
// makes both effects unattributable.
//
// If a more accurate sigmoid is ever wanted, it belongs in `fast_exp.cuh`
// where attention and the MoE would get it too, with its own re-derivation of
// everything calibrated against it.
#include "../fast_exp.cuh"

namespace gr_hyper {

constexpr int THREADS = 256;

__device__ __forceinline__ float gr_sigmoid(float x) {
    return fast_exp::sigmoid<float>(x);
}

__device__ __forceinline__ float4 gr_sigmoid4(float4 x) {
    return fast_exp::sigmoid4<float>(x);
}

// `float4` vector width for a row of `d`, or 0 when the vector path would be
// misaligned.
//
// Every operand here is indexed at a multiple of `d` (stream `s` of row `t`
// starts at `(t·hc + s)·d`), so a `d` that is not a multiple of four puts the
// odd streams on a 4-byte boundary and a `float4` load there faults with
// `CUDA_ERROR_MISALIGNED_ADDRESS` — not a wrong answer, a hard fault, and only
// on the streams above the first. The base pointers themselves are allocation
// aligned: the Rust side asserts `start_offset == 0` (`expect_dense`), so an
// offset view cannot smuggle in a misaligned base either.
//
// The released checkpoint's `d` is 2560, so production always vectorises; the
// scalar loops below are what keep the kernels correct for the widths the
// parity tests deliberately include.
//
// `vec_ok` carries the half of that decision the kernel cannot see. Operands
// reach these launches as **views**, and a view's base is the storage pointer
// advanced by its start offset — so a dense tensor starting at element 1 is
// 4-byte aligned no matter how well-behaved `d` is. The host ANDs the offset
// alignment of every operand into this flag; the kernel keeps the width test,
// because both have to hold.
__device__ __forceinline__ int gr_vec_width(int d, int vec_ok) {
    return (vec_ok != 0 && (d & 3) == 0) ? (d >> 2) : 0;
}

// Block-wide sum over `THREADS` lanes, warp-shuffle then one shared round.
__device__ __forceinline__ float block_sum(float v, float* shared) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    constexpr int WARPS = THREADS / 32;
    float total = 0.f;
    if (threadIdx.x < WARPS) total = shared[threadIdx.x];
    if (warp == 0) {
        #pragma unroll
        for (int off = WARPS / 2; off > 0; off >>= 1) {
            total += __shfl_down_sync(0xffffffffu, total, off);
        }
        if (lane == 0) shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

// ── gr_norm ────────────────────────────────────────────────────────────────
// One block per (row, stream). The reduction is over `d` — the stream's own
// width — which is what makes this a GROUPED norm: scaling one stream cannot
// move another's normed value.
//
//   rsqrt = 1 / sqrt( mean_j(x[t,s,j]^2) + eps )        (per stream)
//   xn[t, s*d + j] = x[t,s,j] * rsqrt * gain[s*d + j]
//
// The `[hc*d]` gain is per (stream, column), which is why the existing
// `rms_norm` kernel cannot carry it: that one's alpha is per-column only, so
// the eager path had to follow it with a second full-tensor broadcast multiply.
// Vectorised `float4` throughout — `d` is 2560 on the released checkpoint, a
// multiple of 4; the scalar tail below keeps the kernel correct for widths
// that are not.
extern "C" __global__ void __launch_bounds__(THREADS) gr_norm_kernel(
    const float* __restrict__ x,     // [n, hc, d]
    const float* __restrict__ gain,  // [hc * d]
    float* __restrict__ xn,          // [n, hc * d]
    int d,
    int hc,
    float eps,
    int vec_ok
) {
    const int row = (int)blockIdx.x / hc;
    const int s = (int)blockIdx.x - row * hc;
    const long long base = ((long long)row * hc + s) * d;
    const float* xs = x + base;
    const float* gs = gain + (long long)s * d;
    float* out = xn + base;

    const int vec = gr_vec_width(d, vec_ok);
    float acc = 0.f;
    const float4* x4 = reinterpret_cast<const float4*>(xs);
    for (int i = threadIdx.x; i < vec; i += THREADS) {
        const float4 v = x4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = (vec << 2) + (int)threadIdx.x; i < d; i += THREADS) {
        const float v = xs[i];
        acc += v * v;
    }

    __shared__ float red[THREADS / 32];
    const float total = block_sum(acc, red);
    // `sum · (1/d)`, not `sum / d`: the reference's `rms_norm` precomputes the
    // reciprocal and multiplies, and under this archive's `--use_fast_math` a
    // division is the approximate one. Same `rsqrtf`, same argument, so the
    // scale matches the path this replaces.
    const float inv_d = 1.0f / (float)d;
    const float rs = rsqrtf(total * inv_d + eps);

    const float4* g4 = reinterpret_cast<const float4*>(gs);
    float4* o4 = reinterpret_cast<float4*>(out);
    for (int i = threadIdx.x; i < vec; i += THREADS) {
        const float4 v = x4[i];
        const float4 g = g4[i];
        float4 o;
        o.x = v.x * rs * g.x;
        o.y = v.y * rs * g.y;
        o.z = v.z * rs * g.z;
        o.w = v.w * rs * g.w;
        o4[i] = o;
    }
    for (int i = (vec << 2) + (int)threadIdx.x; i < d; i += THREADS) {
        out[i] = xs[i] * rs * gs[i];
    }
}

// ── gr_mix ─────────────────────────────────────────────────────────────────
// One block per row. Each thread owns a column slice and walks the `hc`
// streams, so the stream axis is a register accumulation rather than the
// middle-axis reduction that a `sum(1)` would take (measured at ~9.6 ms a call
// on the eager path — the strided-reduce trap this model already paid for
// once).
//
//   mixed[t, j] = (1/hc) · Σ_s xn[t, s*d + j] · sigmoid(gate_raw[t, s*d + j])
//
// The sigmoid is applied here rather than by a separate launch over the wide
// buffer: `gate_raw` arrives straight from the up-projection GEMM.
extern "C" __global__ void __launch_bounds__(THREADS) gr_mix_kernel(
    const float* __restrict__ xn,        // [n, hc * d]
    const float* __restrict__ gate_raw,  // [n, hc * d]
    float* __restrict__ mixed,           // [n, d]
    int d,
    int hc,
    int vec_ok
) {
    const int row = (int)blockIdx.x;
    const long long rbase = (long long)row * hc * d;
    const float inv_hc = 1.0f / (float)hc;
    const int vec = gr_vec_width(d, vec_ok);

    float4* m4 = reinterpret_cast<float4*>(mixed + (long long)row * d);
    for (int i = threadIdx.x; i < vec; i += THREADS) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int s = 0; s < hc; ++s) {
            const long long off = rbase + (long long)s * d;
            const float4 v = reinterpret_cast<const float4*>(xn + off)[i];
            const float4 g = gr_sigmoid4(reinterpret_cast<const float4*>(gate_raw + off)[i]);
            // Accumulated in ascending stream order, which is the order the
            // eager path's `hc − 1` pairwise adds produce: `0 + s0` is exact,
            // so `(((0+s0)+s1)+s2)+s3` is `((s0+s1)+s2)+s3`.
            acc.x += v.x * g.x;
            acc.y += v.y * g.y;
            acc.z += v.z * g.z;
            acc.w += v.w * g.w;
        }
        acc.x *= inv_hc;
        acc.y *= inv_hc;
        acc.z *= inv_hc;
        acc.w *= inv_hc;
        m4[i] = acc;
    }
    for (int j = (vec << 2) + (int)threadIdx.x; j < d; j += THREADS) {
        float acc = 0.f;
        for (int s = 0; s < hc; ++s) {
            const long long off = rbase + (long long)s * d + j;
            acc += xn[off] * gr_sigmoid(gate_raw[off]);
        }
        mixed[(long long)row * d + j] = acc * inv_hc;
    }
}

// ── gr_combine ─────────────────────────────────────────────────────────────
// One block per row, writing a FRESH residual.
//
//   out[t, s, j] = res[t, s, j] + block_out[t, j] · 2·sigmoid(inject[t, s] / hc)
//
// A fresh output rather than an in-place update, for two reasons. The caller's
// `res` is the live residual stream and may be aliased by anything holding a
// clone of that tensor, so mutating it through a raw pointer would be unsound
// in a way nothing would catch; and updating in place would need the caller to
// copy first, which is a whole extra pass over the wide buffer — the copy plus
// the update reads and writes it twice, where this reads once and writes once.
//
// The scatter weight is centred on 1 (`2·sigmoid(0) == 1`), so a zero
// injection is exactly a plain residual add on every stream — the property the
// reference's own test pins.
//
// `inject` is `[n, hc]` with `hc ≤ GR_MAX_HC`; the weights are computed once
// per row into registers rather than re-derived per column.
#define GR_MAX_HC 16

extern "C" __global__ void __launch_bounds__(THREADS) gr_combine_kernel(
    const float* __restrict__ res,       // [n, hc, d]
    const float* __restrict__ block_out, // [n, d]
    const float* __restrict__ inject,    // [n, hc]
    float* __restrict__ out,             // [n, hc, d]
    int d,
    int hc,
    int vec_ok
) {
    const int row = (int)blockIdx.x;
    const float inv_hc = 1.0f / (float)hc;

    float w[GR_MAX_HC];
    for (int s = 0; s < hc; ++s) {
        w[s] = 2.0f * gr_sigmoid(inject[(long long)row * hc + s] * inv_hc);
    }

    const int vec = gr_vec_width(d, vec_ok);
    const float* out_row = block_out + (long long)row * d;
    for (int i = threadIdx.x; i < vec; i += THREADS) {
        const float4 o = reinterpret_cast<const float4*>(out_row)[i];
        for (int s = 0; s < hc; ++s) {
            const long long off = ((long long)row * hc + s) * d;
            const float4 r = reinterpret_cast<const float4*>(res + off)[i];
            float4 v;
            v.x = r.x + o.x * w[s];
            v.y = r.y + o.y * w[s];
            v.z = r.z + o.z * w[s];
            v.w = r.w + o.w * w[s];
            reinterpret_cast<float4*>(out + off)[i] = v;
        }
    }
    for (int j = (vec << 2) + (int)threadIdx.x; j < d; j += THREADS) {
        const float o = out_row[j];
        for (int s = 0; s < hc; ++s) {
            const long long off = ((long long)row * hc + s) * d + j;
            out[off] = res[off] + o * w[s];
        }
    }
}

} // namespace gr_hyper

extern "C" void run_gr_norm(
    const float* x, const float* gain, float* xn,
    int32_t n, int32_t hc, int32_t d, float eps, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0) return;
    gr_hyper::gr_norm_kernel<<<(unsigned)(n * hc), gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        x, gain, xn, d, hc, eps, vec_ok);
}

extern "C" void run_gr_mix(
    const float* xn, const float* gate_raw, float* mixed,
    int32_t n, int32_t hc, int32_t d, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0) return;
    gr_hyper::gr_mix_kernel<<<(unsigned)n, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        xn, gate_raw, mixed, d, hc, vec_ok);
}

extern "C" void run_gr_combine(
    const float* res, const float* block_out, const float* inject, float* out,
    int32_t n, int32_t hc, int32_t d, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0 || hc > GR_MAX_HC) return;
    gr_hyper::gr_combine_kernel<<<(unsigned)n, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        res, block_out, inject, out, d, hc, vec_ok);
}
