// =============================================================================
// Compressor group pool: per-channel softmax over the pooling axis + weighted sum
// =============================================================================
// `Compressor::pool_and_norm` turns a group's pool inputs into one corpus entry:
//
//     entry[g, k] = Σ_p softmax_p(score[g, :, k]) · kv[g, p, k]
//
// Eagerly that is SEVEN launches over a trivial amount of arithmetic — candle's
// `softmax` is the unfused chain (`max_keepdim`, `broadcast_sub`, `exp`,
// `sum_keepdim`, `broadcast_div`) because the reduction is over a NON-last dim,
// then a `broadcast_mul` and a `sum`. Every compressor emit in the model runs
// this: the prefill fleet's pool, the decode wave's per-layer pool, the streamed
// reference and the seal `close` all funnel through the one call site, so a
// single fused kernel covers all of them.
//
// The softmax is PER CHANNEL: it reduces over the `P` pooling rows independently
// for each of the `D` channels, which is what makes the layout so friendly here —
// consecutive threads take consecutive channels, so every load of a `[P, D]`
// plane is fully coalesced, and the reduction each thread owns is a plain
// sequential walk down its own column with no cross-thread communication at all.
// No shared memory, no barriers, no block reduction.
//
// `P` spans a wide range — 8 for the overlapping CSA compressor (2·ratio), up to
// `ratio` = 128 for HCA, and `r + n` for a partial group closed at a turn seal —
// so nothing here may be sized by it. The two passes re-read `score` instead of
// stashing `exp` in a per-thread array: a `float e[P]` with runtime `P` spills to
// local memory and turns every access into a memory op, which is exactly what
// made two of the mHC fusion attempts SLOWER than the launches they replaced.
// Two is the floor without an online (running-max) softmax, which would reorder
// the accumulation; at the widest shape the kernel is bandwidth-bound, so each
// avoided pass over `score` is worth a proportional share of the runtime.
//
// Numerics: this is NOT bit-exact to the eager chain, and measurement says that
// is because the eager chain is the inaccurate one. `pool_kernel_matches_eager`
// compares three ways — kernel, eager device chain, and the same expression in
// plain host f32 — and the kernel tracks the host result ~3 ORDERS OF MAGNITUDE
// more closely than the eager chain does (e.g. 2.5e-5 vs 3.8e-2 max relative at
// [3, 8, 512]). The archive compiles `--use_fast_math`, so the eager path pays
// `__expf` plus an approximate division across FIVE separate kernels, each
// rounding its intermediate out to memory and reading it back; fusing collapses
// all of that into one pass that keeps the intermediates in registers. So the
// gate asserts the kernel is at least as faithful to exact arithmetic as what it
// replaces, which is the property that actually matters and is strictly stronger
// than agreeing with the eager chain's own drift.
//
// The division is hoisted out of the accumulation — `(Σ e·kv) / z` rather than
// the eager path's `Σ (e/z)·kv`. That is a deliberate departure from mirroring
// the reference's structure, taken because it wins on BOTH axes the gate cares
// about: one rounded divide instead of P of them, and one fewer pass over
// `score` on a kernel that is bandwidth-bound at the widest shape.
//
// What is NOT fused here is the RMSNorm that follows in `pool_and_norm`. It
// reduces across ALL of `D`, while the launch config deliberately TILES `D`
// across blocks to have any blocks at all at decode width — so folding it in
// would need either a cross-block reduction (another launch, or atomics plus a
// second pass) or a return to one-block-per-group, which `ncu` measured at 0.09%
// SM throughput. The RMSNorm is already a single fused candle op; leaving it as
// its own launch is what buys the tiling, and the tiling is worth more.

#include <cuda_runtime.h>

// ---- pool: per-channel softmax over P, then the weighted sum ---------------
//   pool_kv, pool_score: [G, P, D] f32
//   out:                 [G, D]    f32
// One block per group; threads stride the D axis.
// The inputs are read through their OWN strides rather than assumed packed.
// Callers hand this dim-0 narrows of a fleet-wide pooled block and other honest
// views, and copying them into a packed layout first would be an allocate-plus-
// copy of the whole plane on a per-layer path — precisely the hot-path invariant
// ("teach the consumer to read the layout that exists") that a `contiguous()`
// here would break. Strides are in ELEMENTS, as candle reports them.
extern "C" __global__ void compressor_pool_kernel(
    const float* __restrict__ pool_kv,    // [G, P, D] strided
    const float* __restrict__ pool_score, // [G, P, D] strided
    float* __restrict__ out,              // [G, D] packed (freshly allocated)
    int P,
    int D,
    long long kv_sg, long long kv_sp, long long kv_sd,
    long long sc_sg, long long sc_sp, long long sc_sd)
{
    // grid.y indexes the group, grid.x tiles the channel axis. Tiling channels
    // is what keeps this off ONE SM at decode: a wave typically closes a group
    // for only a handful of sessions, so a grid of one block per group left
    // `ncu` reporting grid (1,1,1), 0.09% SM throughput and 16% warp occupancy —
    // a kernel doing almost no work but taking 4.7 us, because a single block
    // has nothing to overlap its own memory latency with.
    const int g = blockIdx.y;
    const float* kvg = pool_kv + (long long)g * kv_sg;
    const float* scg = pool_score + (long long)g * sc_sg;
    float* outg = out + (size_t)g * (size_t)D;

    const int kstride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < D; k += kstride) {
        const float* sck = scg + (long long)k * sc_sd;
        const float* kvk = kvg + (long long)k * kv_sd;
        // Pass 1 — max over the pooling axis. A group-0 overlap pad is all
        // -INFINITY on the prev half, which is the intended full mask; the curr
        // half is always finite, so `m` is finite and `expf(-inf - m) == 0`.
        float m = -INFINITY;
        for (int p = 0; p < P; ++p) {
            m = fmaxf(m, sck[(long long)p * sc_sp]);
        }
        // Pass 2 — the denominator AND the weighted sum in ONE walk, dividing
        // once at the end. Splitting these to mirror the eager
        // `broadcast_div` → `broadcast_mul` costs a THIRD read of `score`, and
        // at the widest shape (HCA, P = 128) this kernel runs at ~1.18 TB/s —
        // 85-90% of this part's peak, i.e. bandwidth-bound — so a third of the
        // traffic is a third of the runtime. Hoisting the division is also
        // strictly more accurate: one rounded divide instead of P of them.
        float z = 0.0f, acc = 0.0f;
        for (int p = 0; p < P; ++p) {
            const float e = expf(sck[(long long)p * sc_sp] - m);
            z += e;
            acc += e * kvk[(long long)p * kv_sp];
        }
        outg[k] = acc / z;
    }
}

extern "C" void run_compressor_pool(
    const float* pool_kv, const float* pool_score, float* out,
    int groups, int p, int d,
    long long kv_sg, long long kv_sp, long long kv_sd,
    long long sc_sg, long long sc_sp, long long sc_sd,
    void* stream)
{
    if (groups <= 0 || p <= 0 || d <= 0) return;
    // 128 threads (4 warps) per block, and as many blocks along the channel axis
    // as it takes to cover `D`. A wider block would cover `D` = 512 in one, but
    // the binding constraint at decode is BLOCK COUNT, not threads per block:
    // groups is a handful there, so tiling channels is the only source of blocks
    // to spread across SMs and to overlap each other's memory latency.
    int threads = d < 128 ? ((d + 31) / 32) * 32 : 128;
    if (threads <= 0) threads = 32;
    const int tiles = (d + threads - 1) / threads;
    dim3 grid(tiles, groups, 1);
    compressor_pool_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        pool_kv, pool_score, out, p, d,
        kv_sg, kv_sp, kv_sd, sc_sg, sc_sp, sc_sd);
}
