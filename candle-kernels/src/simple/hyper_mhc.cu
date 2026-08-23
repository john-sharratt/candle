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
    float* __restrict__ comb_raw,        // [n, hc, hc] — IN PLACE: emitted sinkhorn-normalized
    float* __restrict__ y,               // [n, d] — the hc_pre weighted residual reduction
    int n,
    int hc,
    int d,
    float eps,
    int sink_iters,
    float sink_eps)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const int hcd = hc * d;
    const int mix_hc = (2 + hc) * hc;

    // ── Σ xf^2 over the row: float4 loads + a warp-shuffle reduction ──
    //
    // Two things the scalar/shared-tree version paid for needlessly. The loads
    // were one float at a time over `hc*d` (7168) elements — `float4` quarters
    // the load instructions and hits the full 128-bit path. And the reduction
    // was a shared-memory tree with `log2(blockDim)` = 8 `__syncthreads`; a warp
    // does its own 32 lanes with shuffles (no barrier at all), leaving ONE
    // barrier to combine the per-warp partials. On a 16-block grid this kernel
    // is latency-bound, so barrier count on the critical path is what matters.
    //
    // This DOES re-associate the sum (different lane→element assignment), so the
    // rsqrt can differ in the last ULPs from the previous kernel. That is sound
    // here — this reduction was never bit-matched to the CPU reference, which
    // computes it through candle's own `sum`; `fused_pre_post_matches_eager`
    // gates it at 2e-3. It stays fully deterministic: fixed order, no atomics.
    const float* xrow = xf + (size_t)row * hcd;
    float local = 0.0f;
    // `float4` needs 16-byte alignment. Device allocations are 256-byte aligned,
    // so the only risk is the per-row base `xf + row*hcd`: it stays 4-float
    // aligned exactly when `hcd % 4 == 0`. Otherwise take `hcd4 = 0` and let the
    // scalar tail below cover the whole row — the same guard `mhc_post_kernel`
    // applies to its own `d`. Without it an odd `hc*d` faults with `misaligned
    // address` on every row past the first.
    const int hcd4 = (hcd & 3) == 0 ? (hcd >> 2) : 0;
    const float4* xrow4 = reinterpret_cast<const float4*>(xrow);
    for (int k = threadIdx.x; k < hcd4; k += blockDim.x) {
        float4 v = xrow4[k];
        local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int k = (hcd4 << 2) + threadIdx.x; k < hcd; k += blockDim.x) {
        float v = xrow[k];
        local += v * v;
    }

    const unsigned full_mask = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(full_mask, local, o);

    __shared__ float warp_sum[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_sum[warp_id] = local;
    __syncthreads();

    __shared__ float rsqrt_sh;
    if (threadIdx.x == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float tot = 0.0f;
        for (int w = 0; w < n_warps; ++w) tot += warp_sum[w];
        float ms = tot / (float)hcd;
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

    // ── Sinkhorn, in the block that produced the matrix ──
    //
    // This was a separate launch (`sinkhorn_f32_kernel`), and it is ONE THREAD
    // PER MATRIX — at decode `n` is a handful of matrices, so it was a ~1-thread
    // kernel costing ~30 us for a few dozen flops on a tiny [hc, hc]. nsys put
    // it at 5.4% of decode GPU time, essentially all launch overhead. This block
    // already wrote the whole row, so the normalization belongs here.
    //
    // The body below is `sinkhorn.cu` VERBATIM — same operations, same order,
    // same eps placement — so the result is bit-identical to the two-launch
    // form; only the launch is gone. `hyper.rs::HyperConnection::sinkhorn`
    // remains the CPU reference for both.
    __syncthreads(); // comb_row is written by the whole block, read below
    if (sink_iters > 0) {
        // The matrix lives in SHARED memory and one WARP owns it: lane i drives
        // row i for row-phases and column i for column-phases.
        //
        // The obvious transcription — one thread running the scalar loop over a
        // `float c[MHC_MAX_HC*MHC_MAX_HC]` local array — measured 45.2 us at
        // 0.66% SM throughput, WORSE than the separate launch it replaced. Two
        // reasons, both fixed here: a 256-float dynamically-indexed local array
        // spills to local memory, so every one of the ~640 divides became a
        // memory op; and 255 threads sat blocked at the barrier while one
        // worked, so the block could not retire.
        //
        // **Bit-exactness is preserved by the ownership split**: the reference
        // sums each row serially over `j` and each column serially over `i`, so
        // giving one lane the whole row (and one lane the whole column) keeps
        // those accumulations in the reference's order. Nothing is reassociated
        // and no atomics are involved. `__syncwarp` — not `__syncthreads` — is
        // what orders the phases, since a single warp does all of it.
        // The matrix lives in SHARED memory and one WARP owns it: lane i drives
        // row i for row-phases and column i for column-phases.
        //
        // **Bit-exactness comes from the ownership split**: the reference sums
        // each row serially over `j` and each column serially over `i`, so
        // giving one lane the whole row (and one lane the whole column) keeps
        // those accumulations in the reference's order. No reassociation, no
        // atomics. `__syncwarp` — not `__syncthreads` — orders the phases, since
        // a single warp does all of it.
        //
        // Measured alternatives on this exact kernel (ncu, n=16, hc=4, iters=20):
        //   one thread over a local `float[MHC_MAX_HC*MHC_MAX_HC]`  45.2 us
        //   registers + `__shfl_sync` column sums                   29.0 us
        //   this (shared memory + one warp)                         22.4 us
        // Both losers share one cause: `hc` is a RUNTIME value, so a per-thread
        // `float[MHC_MAX_HC]`/`[MHC_MAX_HC*MHC_MAX_HC]` cannot stay in registers
        // and spills to local memory, turning every divide into a memory op.
        // Shared memory keeps the matrix addressable without that spill.
        extern __shared__ float smem_dyn[]; // hc*hc floats, one matrix per block
        float* c = smem_dyn;
        const unsigned lane = threadIdx.x;
        // Warp 0 runs the sinkhorn; warps 1+ run the hc_pre reduction below at
        // the SAME time. The two are independent given `pre`/`comb_raw` (both
        // published by the barrier above), so the long serial sinkhorn overlaps
        // real bandwidth work instead of leaving the rest of the block parked.
        if (lane < 32) {
            // All 32 lanes of warp 0 reach here unconditionally, so the mask is
            // known-full; `__activemask()` would return the same value but reads
            // as though divergence were possible.
            const unsigned active = 0xffffffffu;
            // 1. softmax over j (row-wise), then + eps — lane i owns row i.
            if (lane < (unsigned)hc) {
                const int i = lane;
                float m = -1e30f;
                for (int j = 0; j < hc; ++j) m = fmaxf(m, comb_row[i * hc + j]);
                float s = 0.0f;
                for (int j = 0; j < hc; ++j) {
                    float e = expf(comb_row[i * hc + j] - m);
                    c[i * hc + j] = e;
                    s += e;
                }
                for (int j = 0; j < hc; ++j) c[i * hc + j] = c[i * hc + j] / s + sink_eps;
            }
            __syncwarp(active);

            // 2. column-normalize: / (sum_i + eps) — lane j owns column j.
            if (lane < (unsigned)hc) {
                const int j = lane;
                float s = sink_eps;
                for (int i = 0; i < hc; ++i) s += c[i * hc + j];
                for (int i = 0; i < hc; ++i) c[i * hc + j] /= s;
            }
            __syncwarp(active);

            // 3. (iters-1) x [row-normalize / (sum_j + eps), column-normalize].
            for (int it = 0; it < sink_iters - 1; ++it) {
                if (lane < (unsigned)hc) {
                    const int i = lane;
                    float s = sink_eps;
                    for (int j = 0; j < hc; ++j) s += c[i * hc + j];
                    for (int j = 0; j < hc; ++j) c[i * hc + j] /= s;
                }
                __syncwarp(active);
                if (lane < (unsigned)hc) {
                    const int j = lane;
                    float s = sink_eps;
                    for (int i = 0; i < hc; ++i) s += c[i * hc + j];
                    for (int i = 0; i < hc; ++i) c[i * hc + j] /= s;
                }
                __syncwarp(active);
            }

            for (int k = lane; k < hc * hc; k += 32) comb_row[k] = c[k];
        } else {
            // ── hc_pre stage 2, fused: y[k] = Σ_c pre[c] · x[c, k] ──
            //
            // Was `mhc_pre_reduce_kernel`, a separate launch over the same grid
            // (one block per row) reading the same `[hc*d]` row this block has
            // already streamed for the rsqrt — so folding it in saves both the
            // launch and a second pass over a row that is now warm in L2.
            //
            // `x` is `xf` reshaped: [n, hc, d] and [n, hc*d] are the same
            // storage, so `xrow` above already points at this row's `x`.
            const int rwork = blockDim.x - 32;
            float* yrow = y + (size_t)row * d;
            for (int k = threadIdx.x - 32; k < d; k += rwork) {
                float acc = 0.0f;
                for (int cc = 0; cc < hc; ++cc) acc += pre_row[cc] * xrow[(size_t)cc * d + k];
                yrow[k] = acc;
            }
        }
    } else {
        // Sinkhorn disabled: the reduction still has to happen, and with no
        // warp-0 work to overlap it every thread takes part.
        float* yrow = y + (size_t)row * d;
        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            float acc = 0.0f;
            for (int cc = 0; cc < hc; ++cc) acc += pre_row[cc] * xrow[(size_t)cc * d + k];
            yrow[k] = acc;
        }
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
    const int row = blockIdx.x;
    const int j = blockIdx.y;
    if (row >= n || j >= hc) return;
    const float* bo = block_out + (size_t)row * d;
    const float* res = residual + (size_t)row * hc * d;
    const float* prow = post + (size_t)row * hc;
    const float* crow = comb + (size_t)row * hc * hc;
    float* orow_j = out + ((size_t)row * hc + j) * d;

    // **`j` is a block index, not `e / d`.** The flat form paid an integer
    // divide AND modulo per output element — tens of cycles each on a kernel
    // whose real work is one FMA chain — and it launched only `n` blocks, which
    // on a 16-row decode wave left most of the GPU idle. Indexing by (row, j)
    // removes both divides and multiplies the grid by `hc`.
    //
    // `comb`'s column j and `post[j]` are loop-invariant across the whole block
    // but were re-read from global for every element; they are staged once here.
    extern __shared__ float cj[]; // hc floats: comb[i][j] for i in 0..hc
    __shared__ float pj_sh;
    for (int i = threadIdx.x; i < hc; i += blockDim.x) cj[i] = crow[i * hc + j];
    if (threadIdx.x == 0) pj_sh = prow[j];
    __syncthreads();
    const float pj = pj_sh;

    // Vectorized main pass. The accumulation order is unchanged — `post·out`
    // first, then `comb[i]·res[i]` for ascending i, each output element
    // independent — so results are bit-identical to the scalar form.
    // `float4` requires 16-byte alignment, which holds only when `d` is a
    // multiple of 4 (every row base is then 4-float aligned); otherwise the
    // scalar path below covers everything.
    const int d4 = (d & 3) == 0 ? (d >> 2) : 0;
    for (int k4 = threadIdx.x; k4 < d4; k4 += blockDim.x) {
        const float4 b = reinterpret_cast<const float4*>(bo)[k4];
        float4 acc;
        acc.x = pj * b.x;
        acc.y = pj * b.y;
        acc.z = pj * b.z;
        acc.w = pj * b.w;
        for (int i = 0; i < hc; ++i) {
            const float c = cj[i];
            const float4 r = reinterpret_cast<const float4*>(res + (size_t)i * d)[k4];
            acc.x += c * r.x;
            acc.y += c * r.y;
            acc.z += c * r.z;
            acc.w += c * r.w;
        }
        reinterpret_cast<float4*>(orow_j)[k4] = acc;
    }
    for (int k = (d4 << 2) + threadIdx.x; k < d; k += blockDim.x) {
        float acc = pj * bo[k];
        for (int i = 0; i < hc; ++i) acc += cj[i] * res[(size_t)i * d + k];
        orow_j[k] = acc;
    }
}

// ---- hc_head: the final residual-stream reduction -------------------------
// `hc_head` is `hc_pre` WITHOUT the split: one gate vector, no post, no combine
// matrix, no sinkhorn. `fn_w` is [hc, hc*d] so `mixes_raw` is [n, hc] (not
// [n, (2+hc)*hc]) and `scale` is a single value rather than three.
//   rsqrt  = 1 / sqrt(mean(xf^2, -1) + eps)          (per row)
//   g[i]   = sigmoid(mixes_raw[i] * rsqrt * s + base[i]) + eps
//   y[k]   = Σ_i g[i] * x[i, k]
//
// Eager, this was five full passes over [n, hc, d]: `sqr`, the mean reduction,
// the `broadcast_mul` — which also MATERIALISES a whole [n, hc, d] temp, an
// allocate-plus-copy under invariant 2 — and then `sum(2)`, which reduces the
// SECOND-TO-LAST axis and so reads that temp with a `d`-element stride. That
// last one is why the span measured ~75 GB/s on a card that does ~1 TB/s.
// Here the row is streamed once for the rsqrt and once (warm in L2) for the
// reduction, with no temp and one launch.
//
// The gate lives in shared memory rather than being published to global like
// `mhc_pre_gates_kernel`'s `pre`: nothing downstream of `hc_head` needs it.
//
// ── Measured, and deliberately NOT tuned further ──
// ncu at the decode shape (n=16, hc=4, d=1792) via `bench_fused_head_reduce`:
//   duration 5.31 us      DRAM throughput 6.9%     compute (SM) 1.5%
//   grid 16 blocks        waves/SM 0.02            achieved occupancy 16.6%
//   39 registers/thread (block limit 6 — not the constraint)
// Nothing is saturated: it is parallelism-starved, a 16-block grid on a ~100-SM
// card, and wall-clock per call is 8.9 us so ~3.6 us of that is WDDM launch.
// The levers that would help are known — tile `d` across `grid.y` (the extra
// per-tile rms pass is free at 6.9% DRAM), fold the gate into the same thread-0
// section as the rsqrt to drop one of the three `__syncthreads`, and `float4`
// the reduction loop the way the rms loop above already is.
//
// None of them are worth taking. This kernel runs ONCE PER WAVE, not per layer:
// 2.0 ms across an entire [1,4,8,16,1] sweep, ~0.05% of a decode step. Halving
// it would return ~1 ms out of ~27,000. The win here was the fusion itself
// (`fast_sum` 448 launches / 290.2 ms in `deepseek:head_lm` → zero); what is
// left is noise, and tuning it would be optimising what is measurable rather
// than what is costly. Re-open this only if a profile puts the kernel back on
// the critical path — the harness is already there to measure it.
extern "C" __global__ void mhc_head_reduce_kernel(
    const float* __restrict__ xf,        // [n, hc*d]
    const float* __restrict__ mixes_raw, // [n, hc]
    const float* __restrict__ base,      // [hc]
    const float* __restrict__ scale,     // [1]
    float* __restrict__ y,               // [n, d] — fully written
    int n,
    int hc,
    int d,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const int hcd = hc * d;

    // ── Σ xf^2 over the row — float4 loads + a warp-shuffle reduction ──
    // Identical in form to `mhc_pre_gates_kernel`'s, including the
    // re-association: the lane→element assignment differs from candle's `sum`,
    // so the rsqrt can differ in the last ULPs. That is the same latitude the
    // sibling kernel documents, and `fused_head_reduce_matches_eager` gates it
    // at the same 2e-3. Fully deterministic — fixed order, no atomics.
    const float* xrow = xf + (size_t)row * hcd;
    float local = 0.0f;
    // `float4` needs 16-byte alignment; the per-row base `xf + row*hcd` is only
    // 4-float aligned when `hcd % 4 == 0`. Otherwise `hcd4 = 0` hands the whole
    // row to the scalar tail. See the identical guard in `mhc_post_kernel`.
    const int hcd4 = (hcd & 3) == 0 ? (hcd >> 2) : 0;
    const float4* xrow4 = reinterpret_cast<const float4*>(xrow);
    for (int k = threadIdx.x; k < hcd4; k += blockDim.x) {
        float4 v = xrow4[k];
        local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int k = (hcd4 << 2) + threadIdx.x; k < hcd; k += blockDim.x) {
        float v = xrow[k];
        local += v * v;
    }

    const unsigned full_mask = 0xffffffffu;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(full_mask, local, o);

    __shared__ float warp_sum[32];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_sum[warp_id] = local;
    __syncthreads();

    __shared__ float rsqrt_sh;
    if (threadIdx.x == 0) {
        const int n_warps = (blockDim.x + 31) >> 5;
        float tot = 0.0f;
        for (int w = 0; w < n_warps; ++w) tot += warp_sum[w];
        float ms = tot / (float)hcd;
        rsqrt_sh = 1.0f / sqrtf(ms + eps);
    }
    __syncthreads();
    const float rsqrt = rsqrt_sh;

    // ── The gate, into shared: g[i] = sigmoid(m[i]·rsqrt·s + base[i]) + eps ──
    __shared__ float g[MHC_MAX_HC];
    const float s = scale[0];
    const float* mrow = mixes_raw + (size_t)row * hc;
    for (int i = threadIdx.x; i < hc; i += blockDim.x) {
        float m = mrow[i] * rsqrt;
        g[i] = 1.0f / (1.0f + expf(-(m * s + base[i]))) + eps;
    }
    __syncthreads();

    // ── y[k] = Σ_i g[i] · x[i, k] ──
    // `x` is `xf` reshaped: [n, hc, d] and [n, hc*d] are the same storage, so
    // `xrow` already points at this row's `x`. Every thread participates —
    // unlike the sibling kernel there is no warp-0 sinkhorn to overlap.
    float* yrow = y + (size_t)row * d;
    for (int k = threadIdx.x; k < d; k += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < hc; ++i) acc += g[i] * xrow[(size_t)i * d + k];
        yrow[k] = acc;
    }
}

// ---- launchers ------------------------------------------------------------
extern "C" void run_mhc_pre_gates(
    const float* xf, const float* mixes_raw, const float* base, const float* scale,
    float* pre, float* post, float* comb_raw, float* y,
    int n, int hc, int d, float eps, int sink_iters, float sink_eps, void* stream)
{
    if (n <= 0) return;
    // **The block size is load-bearing, not a tuning knob.** It must be a whole
    // number of warps: the rsqrt reduction uses a full-mask `__shfl_down_sync`,
    // which is undefined if the last warp is partial. It must also be > 32, so
    // warps 1+ exist to run the fused reduction while warp 0 sinkhorns.
    constexpr int PRE_GATES_THREADS = 256;
    static_assert(PRE_GATES_THREADS % 32 == 0 && PRE_GATES_THREADS > 32,
                  "whole warps, and at least one warp beyond the sinkhorn warp");
    // Dynamic shared memory holds this block's [hc, hc] combine matrix for the
    // in-kernel sinkhorn (16 floats at hc=4 — trivial, and it keeps the matrix
    // out of local memory, which is what made the first version slow).
    size_t shmem = (size_t)hc * (size_t)hc * sizeof(float);
    mhc_pre_gates_kernel<<<n, PRE_GATES_THREADS, shmem, (cudaStream_t)stream>>>(
        xf, mixes_raw, base, scale, pre, post, comb_raw, y, n, hc, d, eps,
        sink_iters, sink_eps);
}

extern "C" void run_mhc_head_reduce(
    const float* xf, const float* mixes_raw, const float* base, const float* scale,
    float* y, int n, int hc, int d, float eps, void* stream)
{
    if (n <= 0 || hc <= 0) return;
    // Whole warps, for the same reason as `run_mhc_pre_gates`: the rsqrt
    // reduction uses a full-mask `__shfl_down_sync`, undefined on a partial
    // warp. No sinkhorn warp here, so there is no lower bound beyond that, and
    // no dynamic shared memory — the [hc] gate is a fixed `MHC_MAX_HC` array.
    constexpr int HEAD_REDUCE_THREADS = 256;
    static_assert(HEAD_REDUCE_THREADS % 32 == 0, "whole warps for the shuffle reduction");
    mhc_head_reduce_kernel<<<n, HEAD_REDUCE_THREADS, 0, (cudaStream_t)stream>>>(
        xf, mixes_raw, base, scale, y, n, hc, d, eps);
}

extern "C" void run_mhc_post(
    const float* block_out, const float* residual, const float* post, const float* comb,
    float* out, int n, int hc, int d, void* stream)
{
    if (n <= 0 || hc <= 0) return;
    // One block per (row, output-copy j): `hc`x the grid of the flat form, and
    // `j` becomes a block index so the kernel does no integer division.
    dim3 grid(n, hc, 1);
    size_t shmem = (size_t)hc * sizeof(float); // comb's column j
    mhc_post_kernel<<<grid, 256, shmem, (cudaStream_t)stream>>>(
        block_out, residual, post, comb, out, n, hc, d);
}
