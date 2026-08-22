// =============================================================================
// Indexer score reduction: relu · per-head weight, summed over heads, padded
// =============================================================================
// The two-stage selection's precision stage reduces the per-head query·key
// scores to one score per candidate entry:
//
//     out[b, j] = Σ_h relu(scores[b, h, j]) · w[b, h]
//                 + (j < counts[b] ? 0 : -1e30)
//
// Eagerly that is EIGHT launches — `relu`, `broadcast_mul`, `sum` over the head
// axis (a non-last dim, so candle's reduce, not a fused path), then `arange`,
// `broadcast_lt`, `to_dtype` and `affine` to build the padding mask on-device,
// and `broadcast_add` to apply it. Eight is a floor, not an estimate: candle's
// reduce over a non-last dim may itself be more than one launch.
// Both batched selectors run it: the decode wave's
// per-layer `two_stage_select_batched` and prefill's
// `batched_causal_select_device`, which differ only in what the trailing count
// means — a per-session shortlist width in one, a per-token causal visibility
// bound in the other. Same expression, so one kernel.
//
// The `-1e30` is not a stylistic infinity: the mask is added to REAL scores that
// may be negative, and a padding column scoring 0 could otherwise outrank a
// genuinely negative-scoring entry in the top-k that follows. It is reproduced
// here exactly as the eager `affine(1e30, -1e30)` on a {1,0} mask produced it,
// so a masked column lands on the same bits.
//
// Layout: one block per (batch row, candidate tile); threads stride the
// candidate axis and each privately reduces over heads. Consecutive threads take
// consecutive candidates, so every read of a `[h, mm]` plane coalesces, and
// there is no cross-thread communication at all — no shared memory, no barriers.
// Tiling candidates rather than giving each row one block is what keeps this off
// a handful of SMs at decode width, where `big` is only as large as the number
// of sessions that selected this step.
//
// Inputs are read through their OWN strides. `scores` arrives as the output of a
// batched matmul and `w` from a stack, but neither is contracted to be packed,
// and copying them into a packed layout on a per-layer path is the hot-path
// invariant violation this exists to avoid.
//
// NOTE the deliberately narrow scope. `gallery.rs` records a measured -29%
// prefill from fusing this stage's MATMUL and ARGSORT together: both use the
// whole GPU and neither wants to be trapped in one block. This kernel replaces
// only the elementwise reduction BETWEEN them, and leaves cuBLAS and the
// parallel argsort exactly where they are.

#include <cuda_runtime.h>
#include <map>
#include <mutex>

extern "C" __global__ void indexer_score_reduce_kernel(
    const float* __restrict__ scores, // [B, H, M] strided
    const float* __restrict__ w,      // [B, H]    strided
    const unsigned int* __restrict__ counts, // [B] strided, or null for "no padding"
    float* __restrict__ out,          // [B, M] packed
    int H,
    int M,
    long long sc_sb, long long sc_sh, long long sc_sm,
    long long w_sb, long long w_sh,
    long long cnt_s)
{
    const int b = blockIdx.y;
    const float* scb = scores + (long long)b * sc_sb;
    const float* wb = w + (long long)b * w_sb;
    float* outb = out + (size_t)b * (size_t)M;
    // `counts` is read through its own stride like every other input. It is the
    // one input a caller is most likely to hand over as a view of something
    // wider, and assuming stride 1 here would read a neighbour as a visibility
    // bound and silently mis-mask the top-k rather than fail.
    const unsigned int cnt =
        counts ? counts[(long long)b * cnt_s] : (unsigned int)M;

    const int stride = gridDim.x * blockDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < M; j += stride) {
        const float* scj = scb + (long long)j * sc_sm;
        // Accumulated EXACTLY, in f32. Bit-equality with the eager chain is not
        // reachable — candle reduces the head axis with a tree, not a sequential
        // walk, so the summation orders differ and matching it would mean
        // replicating its reduction shape (measured: 3 ULP at H = 64; neither
        // FMA contraction nor rounding order accounts for it). Since the orders
        // cannot agree, the kernel is held to the weaker but more useful bar of
        // being MORE faithful to exact arithmetic than the chain it replaces.
        // That matters here specifically because this result feeds an argsort,
        // where a few ULP reorder near-tied candidates and change which entries
        // the model attends to.
        //
        // It reaches that bar without a double in sight. Each product is split
        // by FMA into its rounded part and an EXACT residual (r*wv == p + e, no
        // rounding lost), and each addition is split the same way by TWO-SUM, so
        // the whole dot product carries its low-order bits to the end. Measured
        // against an exact host reference this is 0 relative error on every
        // gated shape — the same result the double accumulator gave.
        //
        // Two-sum rather than the cheaper Kahan, and the difference is not
        // academic here: Kahan's compensation is only valid while the running
        // sum dominates the addend, and this reduction violates that constantly
        // — `relu` makes every term non-negative while `w` is signed, so the sum
        // cancels toward zero while the individual products stay large. Kahan
        // measured 6.95e-5 against the eager chain's 4.41e-5, i.e. WORSE than
        // what it replaces and a gate failure. Two-sum's error term is exact
        // whichever operand is larger, for two more instructions.
        //
        // The intrinsics are load-bearing, not decoration: this file compiles
        // under --use_fast_math, and a contracted `p = r*wv` folded into the
        // following add would make `e` and the compensation terms algebraically
        // zero, silently deleting the compensation and leaving a plain
        // sequential f32 sum that is WORSE than the eager tree.
        //
        // Why not double: ncu put this kernel at 84.7% FP64 pipeline
        // utilisation with issue slots only 9.8% busy and DRAM at 45% — FP64
        // runs at 1/64 rate on this part, so three FP64 ops per head (two
        // converts and the FMA) cost more than the entire memory stream. The
        // earlier comment here asserted the opposite ("memory-bound, so the
        // wider adds are free"); it was never measured and it was wrong.
        //
        // A single accumulator, deliberately. Four independent partials were
        // tried to break the loop-carried dependency chain and measured FLAT
        // (prefill 105.9 -> 107.4 us), so the H walk is not latency-bound on the
        // accumulate — the extra code bought nothing and is not kept.
        float acc = 0.0f;  // running sum, high part
        float c = 0.0f;    // carried low-order bits the high part could not hold
        for (int h = 0; h < H; ++h) {
            const float s = scj[(long long)h * sc_sh];
            const float r = (s > 0.0f ? s : 0.0f);
            const float wv = wb[(long long)h * w_sh];
            // Exact product: r*wv == p + e, with e recovered by the FMA.
            const float p = __fmul_rn(r, wv);
            c = __fadd_rn(c, __fmaf_rn(r, wv, -p));
            // Exact sum: acc + p == t + err, whichever of the two is larger.
            const float t = __fadd_rn(acc, p);
            const float bp = __fsub_rn(t, acc);
            c = __fadd_rn(
                c,
                __fadd_rn(__fsub_rn(acc, __fsub_rn(t, bp)), __fsub_rn(p, bp)));
            acc = t;
        }
        // Fold the carried low-order terms back into the high part.
        const float sum = __fadd_rn(acc, c);
        // Exactly what `affine(1e30, -1e30)` on a {1,0} mask then `add` gave:
        // a valid column adds 0, a padding column adds -1e30.
        outb[j] = sum + ((unsigned int)j < cnt ? 0.0f : -1e30f);
    }
}

// Per-device occupancy limits, resolved once per device and reused.
//
// `cudaGetDeviceProperties` is far too slow to call per launch, but caching it
// in bare statics was wrong twice over: the lazy init was an unsynchronised
// write of several statics, and a single cache keyed to whichever device
// happened to be current on the first call would hand device 0's limits to
// every other device in a multi-GPU process. Keyed by device id under a mutex
// fixes both. The lock is uncontended tens of nanoseconds against a kernel
// launch measured in microseconds.
struct OccupancyLimits {
    int warps_per_sm;
    int wide;  // narrowest block that can fill an SM's warp slots
    int sms;
};

static OccupancyLimits occupancy_limits()
{
    static std::mutex mu;
    static std::map<int, OccupancyLimits> cache;

    int dev = 0;
    cudaGetDevice(&dev);

    std::lock_guard<std::mutex> guard(mu);
    auto it = cache.find(dev);
    if (it != cache.end()) return it->second;

    cudaDeviceProp prop;
    OccupancyLimits lim;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        // Fall back to one warp per block: always a valid launch, and the only
        // cost is the occupancy this function exists to recover.
        lim.warps_per_sm = 32;
        lim.wide = 32;
        lim.sms = 1;
        cache[dev] = lim;
        return lim;
    }
    const int warps_sm = prop.maxThreadsPerMultiProcessor / 32;
    const int blocks_sm = prop.maxBlocksPerMultiProcessor;
    int per_block = blocks_sm > 0 ? (warps_sm + blocks_sm - 1) / blocks_sm : 1;
    if (per_block < 1) per_block = 1;
    int wide = per_block * 32;
    if (wide < 32) wide = 32;
    if (wide > 256) wide = 256;
    lim.warps_per_sm = warps_sm > 0 ? warps_sm : 32;
    lim.wide = wide;
    lim.sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    cache[dev] = lim;
    return lim;
}

extern "C" void run_indexer_score_reduce(
    const float* scores, const float* w, const unsigned int* counts, float* out,
    int b, int h, int m,
    long long sc_sb, long long sc_sh, long long sc_sm,
    long long w_sb, long long w_sh,
    long long cnt_s,
    void* stream)
{
    // `h == 0` is NOT excluded here. The reduction over zero heads is 0, and the
    // kernel produces exactly that (the H loop simply does not run) plus the
    // mask — which is what the eager path returns too. Bailing instead would
    // leave `out` as the caller allocated it, and the caller allocates
    // uninitialised under hot-path invariant 6, so an early return here is a
    // silent hand-off of whatever was in that VRAM to the argsort downstream.
    // `b` and `m` are different: at zero the tensor has no elements to fill.
    if (b <= 0 || m <= 0 || h < 0) return;
    // Block width is chosen from the DEVICE's own occupancy limits, because the
    // two shapes want opposite things and neither constant is right for both.
    //
    // Narrow blocks maximise block count, which is what the decode shape needs:
    // `b` is only the number of sessions selecting this step, so at 128 threads
    // m = 128 produced exactly ONE tile — grid (1,1,1), 0.31% SM throughput,
    // 15.4 us for an 8192-MAC reduction. But narrow blocks also CAP occupancy,
    // which is what the prefill shape hits: an SM retires a bounded number of
    // resident BLOCKS, so one-warp blocks can never fill its warp slots however
    // many blocks are queued behind them. Measured at b = 512: one warp per
    // block runs 24.85 us, two run 18.63 us — a third of the runtime, for a
    // kernel that is pure streaming and should be bound by DRAM alone.
    //
    // So: widen to the narrowest block that can fill an SM's warp slots
    // (maxThreadsPerMultiProcessor / maxBlocksPerMultiProcessor, rounded to a
    // warp), and only when the grid still has enough blocks left to hand every
    // SM a full complement. Otherwise stay at one warp and spend the parallelism
    // on block count instead.
    //
    // Splitting the H reduction across blocks would parallelise further still,
    // but needs float atomics into `out`, and non-deterministic accumulation
    // order is precisely the defect that made decode irreproducible once
    // already. Not worth it for a reduction this small.
    const OccupancyLimits lim = occupancy_limits();

    int threads = 32;
    for (int w = 64; w <= lim.wide; w <<= 1) {
        // Blocks this width would produce, against the blocks needed to seat
        // `warps_per_sm` warps on every SM at that width.
        const long long have = (long long)b * ((m + w - 1) / w);
        const int per_sm = (lim.warps_per_sm + (w / 32) - 1) / (w / 32);
        const long long need = (long long)lim.sms * per_sm;
        if (have < need) break;
        threads = w;
    }
    const int tiles = (m + threads - 1) / threads;
    dim3 grid(tiles, b, 1);
    indexer_score_reduce_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        scores, w, counts, out, h, m, sc_sb, sc_sh, sc_sm, w_sb, w_sh, cnt_s);
}
