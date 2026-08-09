// =============================================================================
// mHC (manifold-constrained hyper-connection) FUSED pre/post kernels
// =============================================================================
// The mHC `hc_pre`/`hc_post` around every attention and FFN sub-block are a
// long chain of tiny elementwise tensor ops over small `[hc]`/`[hc,hc]` state
// (rms-rsqrt, the sigmoid gate split, the weighted residual reductions). In
// decode that is ~50 launches per layer over a trivial amount of arithmetic —
// pure launch overhead, the same problem `sinkhorn.cu` already solved for the
// combine-matrix normalization. These kernels fuse the rest of `hc_pre`/`hc_post`
// so the whole mHC is a handful of launches per sub-block instead of ~70.
//
// Semantics match `hyper.rs::HyperConnection` EXACTLY (same op order, same eps
// placement) — the Rust `cpu_*` scalar paths are the bit-exact reference and the
// GPU parity test asserts against them.
//
// One BLOCK per row (row = a (batch·seq) position); `n` is small at decode
// (1 row) and moderate at prefill (prompt length). `hc` is the residual-stream
// copy count (a handful); `d` is the model hidden dim.

#include <cuda_runtime.h>

#define MHC_MAX_HC 16

// ---- hc_pre stage 1: rms-rsqrt · gate split -------------------------------
// From the fused-projection output `mixes_raw = fn_w(xf)` [n, mix_hc] and the
// pre-reshape input `xf` [n, hc*d] (for the rms scale), produce the three gate
// tensors. mix_hc = (2 + hc) * hc:  [0,hc)=pre, [hc,2hc)=post, [2hc,·)=comb.
//   rsqrt   = 1 / sqrt(mean(xf^2, -1) + eps)              (per row)
//   m[k]    = mixes_raw[k] * rsqrt
//   pre[i]  = sigmoid(m[i]      * s0 + base[i])       + eps
//   post[i] = 2 * sigmoid(m[hc+i] * s1 + base[hc+i])
//   comb_raw[i*hc+j] = m[2hc + i*hc+j] * s2 + base[2hc + i*hc+j]
// `scale` is [3] = (s0, s1, s2); `base` is [mix_hc]. comb_raw feeds the
// existing sinkhorn kernel; pre/post are final.
extern "C" __global__ void mhc_pre_gates_kernel(
    const float* __restrict__ xf,        // [n, hc*d]
    const float* __restrict__ mixes_raw, // [n, mix_hc]
    const float* __restrict__ base,      // [mix_hc]
    const float* __restrict__ scale,     // [3]
    float* __restrict__ pre,             // [n, hc]
    float* __restrict__ post,            // [n, hc]
    float* __restrict__ comb_raw,        // [n, hc, hc]
    int n,
    int hc,
    int d,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const int hcd = hc * d;
    const int mix_hc = (2 + hc) * hc;

    // ── Parallel reduction of Σ xf^2 over the row (block-stride) ──
    const float* xrow = xf + (size_t)row * hcd;
    float local = 0.0f;
    for (int k = threadIdx.x; k < hcd; k += blockDim.x) {
        float v = xrow[k];
        local += v * v;
    }
    __shared__ float red[256];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    __shared__ float rsqrt_sh;
    if (threadIdx.x == 0) {
        float ms = red[0] / (float)hcd;
        rsqrt_sh = 1.0f / sqrtf(ms + eps);
    }
    __syncthreads();
    const float rsqrt = rsqrt_sh;

    const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    const float* mrow = mixes_raw + (size_t)row * mix_hc;
    float* pre_row = pre + (size_t)row * hc;
    float* post_row = post + (size_t)row * hc;
    float* comb_row = comb_raw + (size_t)row * hc * hc;

    // pre / post gates (hc each).
    for (int i = threadIdx.x; i < hc; i += blockDim.x) {
        float mp = mrow[i] * rsqrt;
        pre_row[i] = 1.0f / (1.0f + expf(-(mp * s0 + base[i]))) + eps;
        float mq = mrow[hc + i] * rsqrt;
        post_row[i] = 2.0f * (1.0f / (1.0f + expf(-(mq * s1 + base[hc + i]))));
    }
    // comb_raw (hc*hc), pre-sinkhorn affine.
    for (int e = threadIdx.x; e < hc * hc; e += blockDim.x) {
        float mc = mrow[2 * hc + e] * rsqrt;
        comb_row[e] = mc * s2 + base[2 * hc + e];
    }
}

// ---- hc_pre stage 2: weighted residual reduction --------------------------
//   y[row, k] = Σ_c pre[row, c] * x[row, c, k]     (c over hc copies)
// x is the pre-reshape residual [n, hc, d] in F32. One block per row, threads
// over the d axis.
extern "C" __global__ void mhc_pre_reduce_kernel(
    const float* __restrict__ x,   // [n, hc, d]
    const float* __restrict__ pre, // [n, hc]
    float* __restrict__ y,         // [n, d]
    int n,
    int hc,
    int d)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const float* xrow = x + (size_t)row * hc * d;
    const float* prow = pre + (size_t)row * hc;
    float* yrow = y + (size_t)row * d;
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) acc += prow[c] * xrow[(size_t)c * d + k];
        yrow[k] = acc;
    }
}

// ---- hc_post: recombine block output with the residual --------------------
//   new[row, j, k] = post[row, j] * block_out[row, k]
//                    + Σ_i comb[row, i, j] * residual[row, i, k]
// comb is the sinkhorn-normalized [n, hc, hc] (rows = input copy i, cols =
// output copy j). One block per row, threads over the (j, k) output plane.
extern "C" __global__ void mhc_post_kernel(
    const float* __restrict__ block_out, // [n, d]
    const float* __restrict__ residual,  // [n, hc, d]
    const float* __restrict__ post,      // [n, hc]
    const float* __restrict__ comb,      // [n, hc, hc]
    float* __restrict__ out,             // [n, hc, d]
    int n,
    int hc,
    int d)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const float* bo = block_out + (size_t)row * d;
    const float* res = residual + (size_t)row * hc * d;
    const float* prow = post + (size_t)row * hc;
    const float* crow = comb + (size_t)row * hc * hc;
    float* orow = out + (size_t)row * hc * d;

    const int total = hc * d;
    for (int e = threadIdx.x; e < total; e += blockDim.x) {
        int j = e / d;
        int k = e - j * d;
        float acc = prow[j] * bo[k];
        for (int i = 0; i < hc; ++i) acc += crow[i * hc + j] * res[(size_t)i * d + k];
        orow[e] = acc;
    }
}

// ---- launchers ------------------------------------------------------------
extern "C" void run_mhc_pre_gates(
    const float* xf, const float* mixes_raw, const float* base, const float* scale,
    float* pre, float* post, float* comb_raw,
    int n, int hc, int d, float eps, void* stream)
{
    if (n <= 0) return;
    mhc_pre_gates_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        xf, mixes_raw, base, scale, pre, post, comb_raw, n, hc, d, eps);
}

extern "C" void run_mhc_pre_reduce(
    const float* x, const float* pre, float* y, int n, int hc, int d, void* stream)
{
    if (n <= 0) return;
    int threads = d < 256 ? ((d + 31) / 32) * 32 : 256;
    if (threads <= 0) threads = 32;
    mhc_pre_reduce_kernel<<<n, threads, 0, (cudaStream_t)stream>>>(x, pre, y, n, hc, d);
}

extern "C" void run_mhc_post(
    const float* block_out, const float* residual, const float* post, const float* comb,
    float* out, int n, int hc, int d, void* stream)
{
    if (n <= 0) return;
    mhc_post_kernel<<<n, 256, 0, (cudaStream_t)stream>>>(
        block_out, residual, post, comb, out, n, hc, d);
}
