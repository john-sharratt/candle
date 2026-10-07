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
//   gr_combine  res[n,hc,d] += block_out[n,d] · 2·sigmoid(inject[n,hc]/hc)
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
// The one q8a128 tile emitter: the two q8 producers below write, from the same
// floats, exactly what `quantize_q8a128_kernel` writes.
#include "../quantize/q8a128_tile.cuh"
// The trigger that lets a programmatic launch behind the norm start early.
#include "../quantized/pdl.cuh"

namespace gr_hyper {

// 128, not 256: at the released `d` of 2560 a row is 640 `float4`s, which is
// exactly five per thread at 128 and two and a half at 256 — where the last
// pass ran with half its warps idle. It also sizes `gr_mix`/`gr_combine`'s
// column chunks so a row is five whole blocks with no tail.
constexpr int THREADS = 128;

// `float4`s per thread that `gr_norm` keeps in registers between its reduction
// and its write, so the second pass never re-fetches the row. Five covers a
// 2560-wide stream exactly (5 × 128 × 4); a wider stream re-reads only the part
// past it.
constexpr int NORM_CACHE = 5;

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
__host__ __device__ __forceinline__ int gr_vec_width(int d, int vec_ok) {
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
//
// Occupancy is register-limited at 75% (55 registers, most of them the held
// row), and deliberately left there: capping it at 10 or 12 blocks per SM
// spills 20–24 bytes a thread, and `ncu` already measures this kernel at 91%
// of DRAM peak, so the headroom a cap could buy is under a tenth.
//
// ── EMIT_Q8: the same pass, also writing `xn`'s q8a128 operand ─────────────────
// A standalone quantize after the norm would re-read the whole `[n, hc·d]`
// buffer the norm had just written. It does not need to: in the
// store pass below the four warps hold `float4`s `[32w, 32w+32)` of chunk `k`,
// which is 128-element tile `w + 4k` of the stream, lane-major — precisely one
// q8a128 tile per warp per chunk, in `quantize_q8a128_kernel`'s own mapping. So
// each warp quantizes the values it is storing, from registers, through the one
// tile emitter (`q8a128_tile.cuh`), and the bytes are the standalone quantize's.
//
// That needs the vector path and `d` a multiple of 128, which makes `i < vec` the
// same for every lane of a warp — a warp is wholly inside the stream or wholly
// past it, so the emitter's full-mask shuffle never runs a partial warp. The
// host refuses anything else; nothing is lost for the released `d` of 2560.
template <bool EMIT_Q8>
__device__ __forceinline__ void gr_norm_body(
    const float* __restrict__ x,     // [n, hc, d]
    const float* __restrict__ gain,  // [hc * d]
    float* __restrict__ xn,          // [n, hc * d]
    uint8_t* __restrict__ q8,        // EMIT_Q8: q8a128 operand for [n, hc * d]
    int d,
    int hc,
    float eps,
    int vec_ok,
    int sum_norm
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
    // The row is held in registers across the reduction, so the write pass
    // below reads only the gain. Unrolled with a guard rather than looped, so
    // `held` is indexed statically and stays in registers.
    float4 held[NORM_CACHE];
    #pragma unroll
    for (int k = 0; k < NORM_CACHE; ++k) {
        const int i = (int)threadIdx.x + k * THREADS;
        held[k] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < vec) {
            const float4 v = x4[i];
            held[k] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    for (int i = (int)threadIdx.x + NORM_CACHE * THREADS; i < vec; i += THREADS) {
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
    // The stream's first q8a128 tile: `base` is a multiple of `d`, and an
    // emitting `d` a multiple of 128.
    const int stream_tile0 = (int)(base >> 7);
    const int lane = (int)threadIdx.x & 31;
    // One store for both paths; the emit rides it with the values still in
    // registers, so the q8 variant adds no load.
    auto store = [&](int i, float4 v, float4 g) {
        float4 o;
        o.x = v.x * rs * g.x;
        o.y = v.y * rs * g.y;
        o.z = v.z * rs * g.z;
        o.w = v.w * rs * g.w;
        o4[i] = o;
        if constexpr (EMIT_Q8) {
            emit_q8a128_tile(q8, stream_tile0 + (i >> 5), lane, o.x, o.y, o.z, o.w, sum_norm);
        }
    };
    #pragma unroll
    for (int k = 0; k < NORM_CACHE; ++k) {
        const int i = (int)threadIdx.x + k * THREADS;
        if (i < vec) {
            store(i, held[k], g4[i]);
        }
    }
    for (int i = (int)threadIdx.x + NORM_CACHE * THREADS; i < vec; i += THREADS) {
        store(i, x4[i], g4[i]);
    }
    if constexpr (!EMIT_Q8) {
        for (int i = (vec << 2) + (int)threadIdx.x; i < d; i += THREADS) {
            out[i] = xs[i] * rs * gs[i];
        }
    }
}

extern "C" __global__ void __launch_bounds__(THREADS) gr_norm_kernel(
    const float* __restrict__ x, const float* __restrict__ gain, float* __restrict__ xn,
    int d, int hc, float eps, int vec_ok
) {
    gr_norm_body<false>(x, gain, xn, nullptr, d, hc, eps, vec_ok, 0);
}

extern "C" __global__ void __launch_bounds__(THREADS) gr_norm_q8_kernel(
    const float* __restrict__ x, const float* __restrict__ gain, float* __restrict__ xn,
    uint8_t* __restrict__ q8, int d, int hc, float eps, int sum_norm
) {
    // Its consumer is the hyper-connection `down`, a programmatic launch
    // (`quantized/pdl.cuh`): let it start prefetching its weights now. It waits for
    // this grid to finish before it reads `q8`.
    pdl_launch_dependents();
    gr_norm_body<true>(x, gain, xn, q8, d, hc, eps, 1, sum_norm);
}

// ── gr_silu_q8 ─────────────────────────────────────────────────────────────
// The read gate's bottleneck activation, quantized for the up projection:
//
//   q8 ← q8a128( silu(proj[t, 0..cols]) )
//
// `proj` is the stacked down-projection's output, `[n, rows_stride]` with the
// gate columns first (the inject columns after them are left alone), read
// through its row stride. The producer for waves wider than the `up` matmul's
// fused loader serves (`HC_FUSED_SILU_MAX_ROWS`): there each of `up`'s row tiles would
// re-quantize its token tile, and one pass here is cheaper than that.
//
// One warp per 128-element tile — tile `t` is row `t / (cols/128)`, columns
// `128·(t mod cols/128) + [0, 128)`. The SiLU is `fast_exp::silu`, the one
// `usilu_f32` calls and the fused loader calls, so every path's quantized values
// are the same bit for bit.
extern "C" __global__ void __launch_bounds__(THREADS) gr_silu_q8_kernel(
    const float* __restrict__ proj,  // [n, row_stride], gate columns 0..cols
    uint8_t* __restrict__ q8,        // q8a128 operand for [n, cols]
    int n,
    int cols,
    int row_stride,
    int sum_norm
) {
    const int tiles_per_row = cols >> 7;
    const long long tile = ((long long)blockIdx.x * THREADS + threadIdx.x) >> 5;
    if (tile >= (long long)n * tiles_per_row) return;  // whole warps exit together
    const int row = (int)(tile / tiles_per_row);
    const int j = (int)(tile - (long long)row * tiles_per_row);
    const int lane = (int)threadIdx.x & 31;
    const float4 v = *reinterpret_cast<const float4*>(
        proj + (long long)row * row_stride + j * 128 + lane * 4);
    emit_q8a128_tile(q8, (int)tile, lane,
                     fast_exp::silu<float>(v.x), fast_exp::silu<float>(v.y),
                     fast_exp::silu<float>(v.z), fast_exp::silu<float>(v.w), sum_norm);
}

// ── gr_mix ─────────────────────────────────────────────────────────────────
// A 2-D grid: `blockIdx.x` is the row, `blockIdx.y` a `THREADS`-wide chunk of
// its columns, and each thread owns exactly one column (one `float4` on the
// vector path). Every lane of every warp is live at the released width, and a
// decode wave of a few rows still launches five blocks per row rather than one.
//
//   mixed[t, j] = (1/hc) · Σ_s xn[t, s*d + j] · sigmoid(gate_raw[t, s*d + j])
//
// The stream axis is a register accumulation rather than the middle-axis
// reduction a `sum(1)` would take (measured at ~9.6 ms a call on the eager path
// — the strided-reduce trap this model already paid for once). `HC` is a
// template parameter so the stream loop unrolls and all `2·HC` loads issue
// before the first multiply.
//
// The sigmoid is applied here rather than by a separate launch over the wide
// buffer: `gate_raw` arrives straight from the up-projection GEMM.
//
// EMIT_Q8: also write `mixed` as the q8a128 operand the block's own projections
// read. On the vector path the 32 lanes of a warp own 32 consecutive `float4`s of
// one row — 128 elements, one q8a128 tile in `quantize_q8a128_kernel`'s mapping —
// so each warp quantizes the values it is storing and the standalone quantize
// that re-read `mixed` is gone. Needs the vector path and `d` a multiple of 128,
// which makes the `col >= vec` exit whole-warp; the host refuses anything else.
template <int HC, bool EMIT_Q8>
__global__ void __launch_bounds__(THREADS) gr_mix_kernel(
    const float* __restrict__ xn,        // [n, hc * d]
    const float* __restrict__ gate_raw,  // [n, hc * d]
    float* __restrict__ mixed,           // [n, d]
    uint8_t* __restrict__ q8,            // EMIT_Q8: q8a128 operand for [n, d]
    int d,
    int vec_ok,
    int sum_norm
) {
    const int row = (int)blockIdx.x;
    const int col = (int)blockIdx.y * THREADS + (int)threadIdx.x;
    const long long rbase = (long long)row * HC * d;
    // Exact: `HC` is a power of two, and the eager path scales by the same.
    constexpr float inv_hc = 1.0f / (float)HC;
    const int vec = gr_vec_width(d, vec_ok);

    if (vec > 0) {
        if (col >= vec) return;
        float4 v[HC];
        float4 g[HC];
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            const long long off = rbase + (long long)s * d;
            v[s] = reinterpret_cast<const float4*>(xn + off)[col];
            g[s] = reinterpret_cast<const float4*>(gate_raw + off)[col];
        }
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            const float4 gs = gr_sigmoid4(g[s]);
            // Accumulated in ascending stream order, which is the order the
            // eager path's `hc − 1` pairwise adds produce: `0 + s0` is exact,
            // so `(((0+s0)+s1)+s2)+s3` is `((s0+s1)+s2)+s3`.
            acc.x += v[s].x * gs.x;
            acc.y += v[s].y * gs.y;
            acc.z += v[s].z * gs.z;
            acc.w += v[s].w * gs.w;
        }
        acc.x *= inv_hc;
        acc.y *= inv_hc;
        acc.z *= inv_hc;
        acc.w *= inv_hc;
        reinterpret_cast<float4*>(mixed + (long long)row * d)[col] = acc;
        if constexpr (EMIT_Q8) {
            // Row-major tiles of the flat [n, d] operand: row `row`, tile `col / 32`.
            emit_q8a128_tile(q8, row * (d >> 7) + (col >> 5), (int)threadIdx.x & 31,
                             acc.x, acc.y, acc.z, acc.w, sum_norm);
        }
    } else {
        if (col >= d) return;
        float acc = 0.f;
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            const long long off = rbase + (long long)s * d + col;
            acc += xn[off] * gr_sigmoid(gate_raw[off]);
        }
        mixed[(long long)row * d + col] = acc * inv_hc;
    }
}

// ── gr_combine ─────────────────────────────────────────────────────────────
// The same 2-D grid as `gr_mix` — row by column chunk, one column per thread —
// updating the residual IN PLACE.
//
//   res[t, s, j] += block_out[t, j] · 2·sigmoid(inject[t, s] / hc)
//
// In place because the residual is the wave's own buffer, held by the caller
// through `&mut` (the same contract as `Tensor::add_mut`), and a fresh output
// would allocate a whole `[rows, hc, d]` F32 residual twice per layer — 96
// times a forward — for a buffer the next line discards. Each element is read
// and written by the same thread at the same index, so the update needs no
// second buffer; `res` is therefore not `__restrict__`-qualified against
// itself, only against the operands it never aliases.
//
// The scatter weight is centred on 1 (`2·sigmoid(0) == 1`), so a zero
// injection is exactly a plain residual add on every stream — the property the
// reference's own test pins.
//
// `inject` is `[n, hc]` with `hc ≤ GR_MAX_HC`, rows `inject_stride` elements
// apart: it is the tail columns of the pre-mix's stacked down-projection, read
// where that GEMM wrote it rather than compacted first. Every lane of a warp
// reads the same `HC` weights, so those loads are broadcasts.
//
// `HC` is a template parameter so the weights are a statically indexed
// register array. Indexed by a runtime stream count they were not: ptxas put
// them in a 64-byte local-memory stack frame, a round trip per weight per
// column.
//
// GATED: the block output arrives in the MoE's three parts and is assembled here —
//
//   block_out[t, j] = routed[t, j] + shared[t, j] · sigmoid(gate[t])
//
// — the shared expert's per-token sigmoid gate, applied where the residual is
// already being read, instead of as a sigmoid, a broadcast multiply and an add,
// three launches per MoE layer that each round-tripped a `[n, d]` buffer. Same
// operations in the same order as those three, so the same bits.
template <int HC, bool GATED>
__global__ void __launch_bounds__(THREADS) gr_combine_kernel(
    float* res,                          // [n, hc, d], read and written
    const float* __restrict__ block_out, // [n, d] — GATED: the routed experts' sum
    const float* __restrict__ shared,    // GATED: [n, d], the shared expert's output
    const float* __restrict__ gate,      // GATED: [n] raw gate, row stride `gate_stride`
    const float* __restrict__ inject,    // [n, hc], row stride `inject_stride`
    int d,
    int inject_stride,
    int gate_stride,
    int vec_ok
) {
    const int row = (int)blockIdx.x;
    const int col = (int)blockIdx.y * THREADS + (int)threadIdx.x;
    constexpr float inv_hc = 1.0f / (float)HC;
    const int vec = gr_vec_width(d, vec_ok);
    if (col >= (vec > 0 ? vec : d)) return;

    float w[HC];
    #pragma unroll
    for (int s = 0; s < HC; ++s) {
        w[s] = 2.0f * gr_sigmoid(inject[(long long)row * inject_stride + s] * inv_hc);
    }
    // One gate per token, broadcast across its columns.
    float g = 0.f;
    if constexpr (GATED) {
        g = gr_sigmoid(gate[(long long)row * gate_stride]);
    }

    const long long rbase = (long long)row * HC * d;
    if (vec > 0) {
        float4 o = reinterpret_cast<const float4*>(block_out + (long long)row * d)[col];
        if constexpr (GATED) {
            // `__fmul_rn`: the product is rounded before the add, as the separate
            // multiply and add launches rounded it — a contracted FMA would not be.
            const float4 sh = reinterpret_cast<const float4*>(shared + (long long)row * d)[col];
            o.x = o.x + __fmul_rn(sh.x, g);
            o.y = o.y + __fmul_rn(sh.y, g);
            o.z = o.z + __fmul_rn(sh.z, g);
            o.w = o.w + __fmul_rn(sh.w, g);
        }
        float4 v[HC];
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            v[s] = reinterpret_cast<const float4*>(res + rbase + (long long)s * d)[col];
        }
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            v[s].x += o.x * w[s];
            v[s].y += o.y * w[s];
            v[s].z += o.z * w[s];
            v[s].w += o.w * w[s];
            reinterpret_cast<float4*>(res + rbase + (long long)s * d)[col] = v[s];
        }
    } else {
        float o = block_out[(long long)row * d + col];
        if constexpr (GATED) {
            o = o + __fmul_rn(shared[(long long)row * d + col], g);
        }
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            res[rbase + (long long)s * d + col] += o * w[s];
        }
    }
}

} // namespace gr_hyper

// The stream counts `gr_mix` and `gr_combine` are instantiated for — powers of
// two up to `GR_MAX_HC`, the only counts the host lets through (a count that is
// not a power of two would also make the gate's `1/hc` fold inexact).
#define GR_MAX_HC 16
#define GR_DISPATCH_HC(hc, LAUNCH) \
    switch (hc) {                  \
        case 1: LAUNCH(1); break;  \
        case 2: LAUNCH(2); break;  \
        case 4: LAUNCH(4); break;  \
        case 8: LAUNCH(8); break;  \
        case 16: LAUNCH(16); break; \
        default: break;            \
    }

// The 2-D grid shared by `gr_mix` and `gr_combine`: rows on x, `THREADS`-wide
// column chunks on y — `float4` columns on the vector path, scalar otherwise.
static dim3 gr_row_chunk_grid(int n, int d, int vec_ok) {
    const int vec = gr_hyper::gr_vec_width(d, vec_ok);
    const int cols = vec > 0 ? vec : d;
    return dim3((unsigned)n, (unsigned)((cols + gr_hyper::THREADS - 1) / gr_hyper::THREADS), 1);
}

extern "C" void run_gr_norm(
    const float* x, const float* gain, float* xn,
    int32_t n, int32_t hc, int32_t d, float eps, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0) return;
    gr_hyper::gr_norm_kernel<<<(unsigned)(n * hc), gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        x, gain, xn, d, hc, eps, vec_ok);
}

// The q8a128-emitting launchers return GR_Q8_LAUNCHED, or GR_Q8_REFUSED for a shape
// they cannot run — a width that is not a multiple of 128 (a partial warp would run
// the tile emitter's shuffle with lanes missing) or an unsupported `hc`. A refusal
// writes nothing, so the caller must treat it as an error: the operand is consumed by
// the next GEMM, and an unwritten one is wrong numbers, not a fault. Alignment is the
// host's to check (`qwen4exp::hyper::cuda_fused`): a base pointer's offset is not
// visible from here. `n == 0` launches nothing and is not a refusal.
#define GR_Q8_LAUNCHED 0
#define GR_Q8_REFUSED  1

extern "C" int32_t run_gr_norm_q8(
    const float* x, const float* gain, float* xn, void* q8,
    int32_t n, int32_t hc, int32_t d, float eps, int32_t sum_norm, void* stream
) {
    if (n < 0 || hc <= 0 || d <= 0 || (d % 128) != 0) return GR_Q8_REFUSED;
    if (n == 0) return GR_Q8_LAUNCHED;
    gr_hyper::gr_norm_q8_kernel<<<(unsigned)(n * hc), gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        x, gain, xn, reinterpret_cast<uint8_t*>(q8), d, hc, eps, sum_norm);
    return GR_Q8_LAUNCHED;
}

// `cols` a multiple of 128; `row_stride` and `proj` 16-byte aligned (the host's check).
extern "C" int32_t run_gr_silu_q8(
    const float* proj, void* q8,
    int32_t n, int32_t cols, int32_t row_stride, int32_t sum_norm, void* stream
) {
    if (n < 0 || cols <= 0 || (cols % 128) != 0 || row_stride < cols) return GR_Q8_REFUSED;
    if (n == 0) return GR_Q8_LAUNCHED;
    const long long threads = (long long)n * (cols / 128) * 32;
    const unsigned blocks = (unsigned)((threads + gr_hyper::THREADS - 1) / gr_hyper::THREADS);
    gr_hyper::gr_silu_q8_kernel<<<blocks, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>(
        proj, reinterpret_cast<uint8_t*>(q8), n, cols, row_stride, sum_norm);
    return GR_Q8_LAUNCHED;
}

extern "C" void run_gr_mix(
    const float* xn, const float* gate_raw, float* mixed,
    int32_t n, int32_t hc, int32_t d, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0) return;
    const dim3 grid = gr_row_chunk_grid(n, d, vec_ok);
#define GR_LAUNCH_MIX(HC)                                                                     \
    gr_hyper::gr_mix_kernel<HC, false><<<grid, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>( \
        xn, gate_raw, mixed, nullptr, d, vec_ok, 0)
    GR_DISPATCH_HC(hc, GR_LAUNCH_MIX)
#undef GR_LAUNCH_MIX
}

// `gr_mix` also writing `mixed`'s q8a128 operand. `d` a multiple of 128 and `hc` one
// the dispatch instantiates; every operand 16-byte aligned (the vector path — the
// host's check).
extern "C" int32_t run_gr_mix_q8(
    const float* xn, const float* gate_raw, float* mixed, void* q8,
    int32_t n, int32_t hc, int32_t d, int32_t sum_norm, void* stream
) {
    const bool hc_ok = hc == 1 || hc == 2 || hc == 4 || hc == 8 || hc == 16;
    if (n < 0 || !hc_ok || d <= 0 || (d % 128) != 0) return GR_Q8_REFUSED;
    if (n == 0) return GR_Q8_LAUNCHED;
    const dim3 grid = gr_row_chunk_grid(n, d, 1);
#define GR_LAUNCH_MIX_Q8(HC)                                                                 \
    gr_hyper::gr_mix_kernel<HC, true><<<grid, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>( \
        xn, gate_raw, mixed, reinterpret_cast<uint8_t*>(q8), d, 1, sum_norm)
    GR_DISPATCH_HC(hc, GR_LAUNCH_MIX_Q8)
#undef GR_LAUNCH_MIX_Q8
    return GR_Q8_LAUNCHED;
}

extern "C" void run_gr_combine(
    float* res, const float* block_out, const float* inject,
    int32_t n, int32_t hc, int32_t d, int32_t inject_stride, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0 || inject_stride < hc) return;
    const dim3 grid = gr_row_chunk_grid(n, d, vec_ok);
#define GR_LAUNCH_COMBINE(HC)                                                                     \
    gr_hyper::gr_combine_kernel<HC, false><<<grid, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>( \
        res, block_out, nullptr, nullptr, inject, d, inject_stride, 0, vec_ok)
    GR_DISPATCH_HC(hc, GR_LAUNCH_COMBINE)
#undef GR_LAUNCH_COMBINE
}

// The MoE's combine with its block output assembled in place:
// `res += (routed + shared · sigmoid(gate)) · 2·sigmoid(inject/hc)`.
extern "C" void run_gr_combine_gated(
    float* res, const float* routed, const float* shared, const float* gate, const float* inject,
    int32_t n, int32_t hc, int32_t d, int32_t inject_stride, int32_t gate_stride,
    int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0 || inject_stride < hc || gate_stride < 1) return;
    const dim3 grid = gr_row_chunk_grid(n, d, vec_ok);
#define GR_LAUNCH_COMBINE_GATED(HC)                                                              \
    gr_hyper::gr_combine_kernel<HC, true><<<grid, gr_hyper::THREADS, 0, (cudaStream_t)stream>>>( \
        res, routed, shared, gate, inject, d, inject_stride, gate_stride, vec_ok)
    GR_DISPATCH_HC(hc, GR_LAUNCH_COMBINE_GATED)
#undef GR_LAUNCH_COMBINE_GATED
}
