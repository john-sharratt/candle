// =============================================================================
// Compressor group pool: per-channel softmax over the pooling axis + weighted sum
// =============================================================================
// `Compressor::pool_and_norm` turns a group's pool inputs into one corpus entry:
//
//     entry[g, k] = Σ_p softmax_p(score[g, p, k] + ape[p, k]) · kv[g, p, k]
//
// Eagerly that is EIGHT launches over a trivial amount of arithmetic — candle's
// `softmax` is the unfused chain (`max_keepdim`, `broadcast_sub`, `exp`,
// `sum_keepdim`, `broadcast_div`) because the reduction is over a NON-last dim,
// then a `broadcast_mul` and a `sum`, on top of the `broadcast_add` that applies
// the absolute positional encoding. Every compressor emit in the model runs
// this: the prefill fleet's pool, the decode wave's per-layer pool, the streamed
// reference and the seal `close` all funnel through the one call site, so a
// single fused kernel covers all of them.
//
// The softmax is PER CHANNEL: it reduces over the pooling rows independently
// for each of the `D` channels, which is what makes the layout so friendly here —
// consecutive threads take consecutive channels, so every load of a row is fully
// coalesced, and the reduction each thread owns is a plain sequential walk down
// its own column with no cross-thread communication at all. No shared memory, no
// barriers, no block reduction.
//
// ---- The descriptor table ---------------------------------------------------
//
// A group's pooling rows do NOT live in one dense block, and requiring that they
// did was this kernel's API bug. Three separate producers had to concatenate:
//
//   * the overlapping compressor pools the PREVIOUS group's first-half channels
//     followed by the CURRENT group's second-half channels — two column windows
//     of two different tensors;
//   * the decode streamer holds a group's `ratio` rows as `ratio` INDEPENDENT
//     one-row tensors, because each arrived on a different step from a different
//     batched projection, and it must also retain them as the next group's
//     previous half;
//   * the wave pools groups belonging to many independent sessions in one
//     launch, so every session brings its own base addresses.
//
// A single-base-pointer signature forces all three to `cat` their pieces into
// the layout the kernel wanted — an allocate-plus-copy per group per compressor
// per layer, measurably the largest `ucopy_f32` population in the model.
//
// So the kernel takes a DEVICE TABLE instead (hot-path invariant 2b). A group is
// a list of SEGMENTS; a segment is a run of `rows` rows read in place through its
// own base pointer and row stride. The group's pooling axis is segment 0's rows,
// then segment 1's, and so on — exactly the order the concatenation produced, so
// the accumulation is unchanged. A dense producer emits one or two wide segments;
// the streamer emits one segment per row. Because row counts are per segment
// rather than a launch-wide `P`, RAGGED groups come for free: a turn seal closes
// a partial group of `r + n` rows alongside full `2r` ones with no special case,
// and a group whose "previous group" does not exist simply omits that segment
// (its all-`−∞` pad contributed nothing to either pass — an omitted
// `expf(−∞ − m) == 0` term is the same number, not an approximation of one).
//
// `ape` rides in the same table. It is a per-ROW-of-the-group bias on the score,
// shared by every group, so a segment carries its own `ape` base + row stride and
// the bias is applied as the score is read. That removes the `broadcast_add` AND
// removes the reason the producers had to keep an `ape`-added copy of their rows
// at all — they now retain the raw projections, which is what lets the streamer
// hold a group as `ratio` untouched row views.
//
// Table layout, one i64 array, two parts:
//
//   part 1, groups, struct-of-arrays with stride G, so the two loads a block
//   issues are broadcast across its threads:
//       desc[0·G + g]  first segment index of group g
//       desc[1·G + g]  segment count of group g
//
//   part 2, segments, array-of-structs at `desc + 2·G`, POOL_SEG_WORDS each
//   (one 56-byte run per segment, read once per block):
//       [0] kv base pointer        [1] kv row stride
//       [2] score base pointer     [3] score row stride
//       [4] ape base pointer (0 = none)
//       [5] ape row stride         [6] row count
//
// Strides are in ELEMENTS, as candle reports them. The channel stride is 1 in
// every operand by construction — every producer hands row-major rows, and the
// host asserts it — so a channel index adds directly to a row address.
//
// ---- Numerics ---------------------------------------------------------------
//
// This is NOT bit-exact to the eager chain, and measurement says that is because
// the eager chain is the inaccurate one. `pool_kernel_matches_eager` compares
// three ways — kernel, eager device chain, and the same expression in plain host
// f32 — and the kernel tracks the host result ~3 ORDERS OF MAGNITUDE more
// closely than the eager chain does (e.g. 2.5e-5 vs 3.8e-2 max relative at
// [3, 8, 512]). The archive compiles `--use_fast_math`, so the eager path pays
// `__expf` plus an approximate division across FIVE separate kernels, each
// rounding its intermediate out to memory and reading it back; fusing collapses
// all of that into one pass that keeps the intermediates in registers. So the
// gate asserts the kernel is at least as faithful to exact arithmetic as what it
// replaces, which is the property that actually matters and is strictly stronger
// than agreeing with the eager chain's own drift.
//
// Folding `ape` in does NOT change the score's value: it was a single rounded
// f32 add before, materialised to memory, and it is a single rounded f32 add
// now, kept in a register. `__fadd_rn` spells it explicitly so that
// `--use_fast_math` cannot reassociate `(score + ape) − m`, which would change
// the number the exponential sees.
//
// The two passes re-read `score` instead of stashing `exp` in a per-thread array:
// a `float e[P]` with runtime `P` spills to local memory and turns every access
// into a memory op, which is exactly what made two of the mHC fusion attempts
// SLOWER than the launches they replaced. Two is the floor without an online
// (running-max) softmax, which would reorder the accumulation.
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

// ---- float4 is a MEASURED DEAD END here; do not re-attempt --------------------
//
// `ncu` at the prefill-fleet shape (256 groups × 8 rows × 512 channels) puts this
// kernel at 11.36 us, DRAM 58 % of peak, SM 15.6 %, 52 % of warps active. That
// reads like a memory streamer short of the roofline for want of requests in
// flight, and the textbook lever is `float4` — a thread taking four channels, so
// each warp load is 512 B instead of 128 B.
//
// It was implemented (templated on `VEC`, with a host-side alignment check) and
// it is SLOWER at every shape: prefill 20.1 → 21.1 us, HCA 19.1 → 33.5 us. The
// reason is arithmetic, not subtle: `d` is FIXED at 512, so four channels per
// thread means a QUARTER of the lanes — 128 instead of 512 — and with 128
// threads per block the channel tiling collapses from 4 blocks per group to 1.
// Trying to win the blocks back by narrowing the block only made it worse:
// `ncu` measured the result at block size 32, grid 64, **2.08 % of warps
// active** and a 62 us kernel, because one warp per block caps occupancy (an SM
// seats a bounded number of resident BLOCKS).
//
// The general rule this kernel is evidence for: vectorising a loop whose trip
// count is fixed by the data trades parallel work items for per-thread work, and
// on a kernel that is short of warps rather than short of bandwidth that trade
// is always backwards. `float4` here would need `d` ≥ 2048 to keep the same warp
// count. The configuration below — 128 threads, channels tiled across blocks —
// is the best measured one.

#include <cuda_runtime.h>

// i64 words per group (part 1) and per segment (part 2) of the descriptor table.
// Mirrored by `POOL_GROUP_WORDS` / `POOL_SEG_WORDS` on the Rust side, which
// builds the table.
#define POOL_GROUP_WORDS 2
#define POOL_SEG_WORDS 7

// One segment's descriptor, unpacked from the table into registers once per
// block. `rows == 0` is legal and contributes nothing.
struct PoolSeg {
    const float* kv;
    long long kv_s;
    const float* sc;
    long long sc_s;
    const float* ape;
    long long ape_s;
    int rows;
};

__device__ __forceinline__ PoolSeg load_seg(const long long* __restrict__ sd)
{
    PoolSeg s;
    s.kv = (const float*)sd[0];
    s.kv_s = sd[1];
    s.sc = (const float*)sd[2];
    s.sc_s = sd[3];
    s.ape = (const float*)sd[4];
    s.ape_s = sd[5];
    s.rows = (int)sd[6];
    return s;
}

// The score this row contributes, `ape` included. `__fadd_rn` keeps the add from
// being reassociated into the `− m` that follows it.
__device__ __forceinline__ float seg_score(const PoolSeg& s, int p, int k)
{
    const float v = s.sc[(long long)p * s.sc_s + k];
    return s.ape ? __fadd_rn(v, s.ape[(long long)p * s.ape_s + k]) : v;
}

// ---- pool: per-channel softmax over the group's rows, then the weighted sum --
//   desc: the table described in the header
//   out:  [G, D] f32 packed (freshly allocated, fully written)
// grid.y indexes the group, grid.x tiles the channel axis.
extern "C" __global__ void compressor_pool_kernel(
    const long long* __restrict__ desc,
    float* __restrict__ out,
    int G,
    int D)
{
    const int g = blockIdx.y;
    const int seg_begin = (int)desc[0 * G + g];
    const int seg_count = (int)desc[1 * G + g];
    const long long* __restrict__ segs =
        desc + POOL_GROUP_WORDS * (long long)G + (long long)seg_begin * POOL_SEG_WORDS;

    float* outg = out + (size_t)g * (size_t)D;

    const int kstride = gridDim.x * blockDim.x;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < D; k += kstride) {
        // Pass 1 — max over the group's rows, segment by segment in order.
        float m = -INFINITY;
        for (int s = 0; s < seg_count; ++s) {
            const PoolSeg seg = load_seg(segs + (long long)s * POOL_SEG_WORDS);
            for (int p = 0; p < seg.rows; ++p) {
                m = fmaxf(m, seg_score(seg, p, k));
            }
        }
        // Pass 2 — the denominator AND the weighted sum in ONE walk, dividing
        // once at the end. Splitting these to mirror the eager
        // `broadcast_div` → `broadcast_mul` costs a THIRD read of `score`, and
        // this kernel is bandwidth-bound at the wide shapes — so a third of the
        // traffic is a third of the runtime. Hoisting the division is also
        // strictly more accurate: one rounded divide instead of P of them.
        float z = 0.0f, acc = 0.0f;
        for (int s = 0; s < seg_count; ++s) {
            const PoolSeg seg = load_seg(segs + (long long)s * POOL_SEG_WORDS);
            for (int p = 0; p < seg.rows; ++p) {
                const float e = expf(seg_score(seg, p, k) - m);
                z += e;
                acc += e * seg.kv[(long long)p * seg.kv_s + k];
            }
        }
        outg[k] = acc / z;
    }
}

extern "C" void run_compressor_pool(
    const long long* desc,
    float* out,
    int groups,
    int d,
    void* stream)
{
    if (groups <= 0 || d <= 0) return;
    // 128 threads (4 warps) per block, and as many blocks along the channel axis
    // as it takes to cover `D`. The binding constraint at decode width is BLOCK
    // COUNT — `groups` is a handful there, so tiling channels is the only source
    // of blocks to spread across SMs and to overlap each other's memory latency.
    // Do NOT narrow the block to manufacture more of them: measured at 32
    // threads the grid reaches every SM and still collapses to 2.08 % of warps
    // active, because an SM seats a bounded number of resident blocks.
    int threads = d < 128 ? ((d + 31) / 32) * 32 : 128;
    if (threads <= 0) threads = 32;
    const int tiles = (d + threads - 1) / threads;
    dim3 grid(tiles, groups, 1);
    compressor_pool_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        desc, out, groups, d);
}
