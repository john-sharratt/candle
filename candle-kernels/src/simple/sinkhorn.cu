// =============================================================================
// SINKHORN (doubly-stochastic) NORMALIZATION — fused single-launch
// =============================================================================
// Replaces the ~120 tiny host-orchestrated tensor ops per call in the mHC
// `sinkhorn` (softmax + an `iters`-long alternating row/col-normalize loop over
// a small `[hc, hc]` combine matrix). In decode this runs for every sub-block
// (hc_attn, hc_ffn) of every layer, so the launch overhead dominated a trivial
// amount of arithmetic. This kernel does the whole normalization for `n`
// matrices in ONE launch.
//
// Semantics match `hyper.rs::HyperConnection::sinkhorn` EXACTLY (same op order,
// same eps placement, dim 2 = row `i`, dim 3 = col `j`):
//   1. softmax over j (per row), then + eps
//   2. column-normalize:  c[i,j] /= (Σ_i c[i,j] + eps)
//   3. repeat (iters-1) times:
//        row-normalize:    c[i,j] /= (Σ_j c[i,j] + eps)
//        column-normalize: c[i,j] /= (Σ_i c[i,j] + eps)
//
// One thread per matrix: `hc` is tiny (the residual-stream copy count, a
// handful), so each matrix's `hc*hc` state lives in registers/local memory and
// the per-thread loop is cheap. No atomics, no shared memory, bit-deterministic.

#include <cuda_runtime.h>

#define SINKHORN_MAX_HC 16

extern "C" __global__ void sinkhorn_f32_kernel(
    const float* __restrict__ inp, // [n, hc, hc] row-major
    float* __restrict__ out,       // [n, hc, hc]
    int n,
    int hc,
    int iters,
    float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int hh = hc * hc;
    const float* a = inp + (size_t)idx * hh;
    float* o = out + (size_t)idx * hh;

    float c[SINKHORN_MAX_HC * SINKHORN_MAX_HC];

    // 1. softmax over j (row-wise), then + eps.
    for (int i = 0; i < hc; ++i) {
        float m = -1e30f;
        for (int j = 0; j < hc; ++j) m = fmaxf(m, a[i * hc + j]);
        float s = 0.0f;
        for (int j = 0; j < hc; ++j) {
            float e = expf(a[i * hc + j] - m);
            c[i * hc + j] = e;
            s += e;
        }
        for (int j = 0; j < hc; ++j) c[i * hc + j] = c[i * hc + j] / s + eps;
    }

    // 2. column-normalize: ÷ (Σ_i + eps).
    for (int j = 0; j < hc; ++j) {
        float s = eps;
        for (int i = 0; i < hc; ++i) s += c[i * hc + j];
        for (int i = 0; i < hc; ++i) c[i * hc + j] /= s;
    }

    // 3. (iters-1) × [row-normalize ÷ (Σ_j + eps), column-normalize ÷ (Σ_i + eps)].
    for (int it = 0; it < iters - 1; ++it) {
        for (int i = 0; i < hc; ++i) {
            float s = eps;
            for (int j = 0; j < hc; ++j) s += c[i * hc + j];
            for (int j = 0; j < hc; ++j) c[i * hc + j] /= s;
        }
        for (int j = 0; j < hc; ++j) {
            float s = eps;
            for (int i = 0; i < hc; ++i) s += c[i * hc + j];
            for (int i = 0; i < hc; ++i) c[i * hc + j] /= s;
        }
    }

    for (int k = 0; k < hh; ++k) o[k] = c[k];
}

extern "C" void run_sinkhorn_f32(
    const float* inp,
    float* out,
    int n,
    int hc,
    int iters,
    float eps,
    void* stream)
{
    if (n <= 0) return;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    sinkhorn_f32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        inp, out, n, hc, iters, eps);
}
