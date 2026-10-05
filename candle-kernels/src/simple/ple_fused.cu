// =============================================================================
// PLE (per-layer n-gram embedding) FUSED GATE / CONV
// =============================================================================
// Qwen3.8-Flash-Next's PLE block runs once per forward, before layer 1's mixer,
// over every row of the wave (`docs/qwen38_flash_next.md` §12.4). Expressed as
// eager ops it is three grouped norms, a per-stream dot product, a five-op
// scalar chain for the gate, a broadcast, a fourth norm, and a dilated causal
// conv spelled as four gathers, four multiplies and three adds — every one of
// them a `[rows, hc·d]` F32 buffer, about 930 KB a row. At a 2,048-row prefill
// that is 1.9 GB of transients for one block.
//
// Three launches replace everything after the key|value GEMM:
//
//   ple_gate     kv[n, hc·d + d], res[n, hc, d]        → res += gated (in place)
//                                                        normalized[n, hc·d]
//                per (row, stream): the key and query grouped norms, their dot,
//                the signed-sqrt sigmoid gate, the gated value and its conv
//                norm, from one read of each operand row.
//   ple_conv     normalized, histories (by descriptor) → res += silu(conv) (in place)
//                the four dilated taps and the SiLU per channel; a tap that
//                reaches before its segment reads its sequence's history
//                through the span table — no concatenation of histories and
//                rows (invariant 2b).
//   ple_history  each sequence's next history into its spare buffer.
//
// The eager chain in `models::qwen4exp::ple` is the reference these are asserted
// against, and it stays as the CPU implementation. The sigmoid is
// `fast_exp::sigmoid<float>`, the one the eager `sigmoid` op launches, for the
// reason `gr_hyper.cu` gives at length: a fusion is the same computation
// faster, not a different one.
//
// Every kernel takes the `float4` path when its widths are multiples of four
// and the host says every base is 16-byte aligned (`vec_ok`), and a scalar
// path otherwise — production always vectorises (`d` = 2560); the scalar path
// keeps the parity tests' odd widths correct.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

#include "../fast_exp.cuh"

namespace ple_fused {

// 128: a 2,560-wide stream is 640 `float4`s, five per thread exactly, so no
// pass runs a partial warp.
constexpr int THREADS = 128;

// `float4`s per thread `ple_gate` keeps in registers between its reduction and
// its write: five covers a 2,560-wide stream exactly (5 × 128 × 4).
constexpr int CACHE = 5;

// Blocks per SM `ple_conv` is bounded to, which caps its registers at
// 65,536 / (THREADS × 12) = 42: full occupancy, no spills, for a body of four
// loads and a sigmoid.
//
// `ple_gate` is deliberately NOT bounded. It holds the query and value rows
// across its reduction — 88 registers, 42% occupancy — and that is the faster
// kernel. Measured on the RTX PRO 5000 at 2,048 rows (`ple_fused_bench` under
// `ncu`): holding both, 265 µs at 87.7% of DRAM peak; bounded to 8 blocks
// (64 registers, 67% occupancy) with the value row re-read from L2, 285 µs at
// 86.4%. The kernel is at the memory floor either way, so occupancy buys
// nothing and the re-read costs a pass.
constexpr int CONV_MIN_BLOCKS = 12;

// A sequence's rows in the wave, and where its conv history lives.
//
//   start     its first row in the wave
//   len       its row count
//   hist      `[hist_rows, hc·d]` — the history this wave reads
//   new_hist  `[hist_rows, hc·d]` — where this wave's history is written; a
//             different buffer from `hist`, so a failed wave leaves the entering
//             history as it was
struct SpanDesc {
    int64_t start;
    int64_t len;
    const float* hist;
    float* new_hist;
};
static_assert(sizeof(SpanDesc) == 32, "SpanDesc is four 64-bit words");

__host__ __device__ __forceinline__ int vec_width(int d, int vec_ok) {
    return (vec_ok != 0 && (d & 3) == 0) ? (d >> 2) : 0;
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 mul4(float4 a, float4 b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ __forceinline__ float4 scale4(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// Block-wide sum of four accumulators at once: one shuffle tree and one shared
// round for all four, where four `block_sum`s would pay four.
__device__ __forceinline__ float4 block_sum4(float4 v, float4* shared) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v.x += __shfl_down_sync(0xffffffffu, v.x, off);
        v.y += __shfl_down_sync(0xffffffffu, v.y, off);
        v.z += __shfl_down_sync(0xffffffffu, v.z, off);
        v.w += __shfl_down_sync(0xffffffffu, v.w, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    constexpr int WARPS = THREADS / 32;
    if (threadIdx.x == 0) {
        float4 t = shared[0];
        #pragma unroll
        for (int w = 1; w < WARPS; ++w) t = add4(t, shared[w]);
        shared[0] = t;
    }
    __syncthreads();
    return shared[0];
}

// ── ple_gate ───────────────────────────────────────────────────────────────
// One block per (row t, stream s). With k = key[t,s,·], q = res[t,s,·] (the
// query is the wide residual before this block writes it) and v = value[t,·]:
//
//   rk = rsqrt(mean(k²) + eps)          rq = rsqrt(mean(q²) + eps)
//   dot = Σ_j (k_j·gk_j)·(q_j·gq_j) · rk · rq · inv_sqrt_d
//   gate = sigmoid( sgn(dot) · sqrt(clamp(|dot|, 1e-6, 1e30)) )
//   gated_j = v_j · gate
//   rg = rsqrt(gate² · mean(v²) + eps)          (= rsqrt(mean(gated²) + eps))
//   res[t,s,j]            = q_j + gated_j
//   normalized[t, s·d+j]  = gated_j · rg · gc_j
//
// The three norms and the dot reduce together in one pass of four sums; the
// row's `q` and `v` stay in registers for the write pass, so it reads only the
// conv gain (see CONV_MIN_BLOCKS for why this kernel is left unbounded).
extern "C" __global__ void __launch_bounds__(THREADS) ple_gate_kernel(
    const float* __restrict__ kv,       // [n, kv_stride]: key [hc·d], then value [d]
    float* __restrict__ res,            // [n, hc, d], read and written
    const float* __restrict__ gk,       // [hc·d] key norm gain
    const float* __restrict__ gq,       // [hc·d] query norm gain
    const float* __restrict__ gc,       // [hc·d] conv-input norm gain
    float* __restrict__ normalized,     // [n, hc·d]
    int hc,
    int d,
    int kv_stride,
    float eps,
    float inv_sqrt_d,
    int vec_ok
) {
    const int row = (int)blockIdx.x / hc;
    const int s = (int)blockIdx.x - row * hc;
    const long long hcd = (long long)hc * d;
    const float* k = kv + (long long)row * kv_stride + (long long)s * d;
    const float* v = kv + (long long)row * kv_stride + hcd;
    float* q = res + (long long)row * hcd + (long long)s * d;
    const float* gks = gk + (long long)s * d;
    const float* gqs = gq + (long long)s * d;
    const float* gcs = gc + (long long)s * d;
    float* out = normalized + (long long)row * hcd + (long long)s * d;
    const int vec = vec_width(d, vec_ok);

    // x: Σk²   y: Σq²   z: Σ k·gk·q·gq   w: Σv²
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 held_q[CACHE];
    float4 held_v[CACHE];
    const float4* k4 = reinterpret_cast<const float4*>(k);
    const float4* q4 = reinterpret_cast<const float4*>(q);
    const float4* v4 = reinterpret_cast<const float4*>(v);
    const float4* gk4 = reinterpret_cast<const float4*>(gks);
    const float4* gq4 = reinterpret_cast<const float4*>(gqs);
    auto fold = [&](float4 kk, float4 qq, float4 vv, float4 a, float4 b) {
        acc.x += kk.x * kk.x + kk.y * kk.y + kk.z * kk.z + kk.w * kk.w;
        acc.y += qq.x * qq.x + qq.y * qq.y + qq.z * qq.z + qq.w * qq.w;
        acc.z += (kk.x * a.x) * (qq.x * b.x) + (kk.y * a.y) * (qq.y * b.y)
               + (kk.z * a.z) * (qq.z * b.z) + (kk.w * a.w) * (qq.w * b.w);
        acc.w += vv.x * vv.x + vv.y * vv.y + vv.z * vv.z + vv.w * vv.w;
    };
    #pragma unroll
    for (int c = 0; c < CACHE; ++c) {
        const int i = (int)threadIdx.x + c * THREADS;
        held_q[c] = make_float4(0.f, 0.f, 0.f, 0.f);
        held_v[c] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < vec) {
            held_q[c] = q4[i];
            held_v[c] = v4[i];
            fold(k4[i], held_q[c], held_v[c], gk4[i], gq4[i]);
        }
    }
    for (int i = (int)threadIdx.x + CACHE * THREADS; i < vec; i += THREADS) {
        fold(k4[i], q4[i], v4[i], gk4[i], gq4[i]);
    }
    for (int j = (vec << 2) + (int)threadIdx.x; j < d; j += THREADS) {
        const float kj = k[j];
        const float qj = q[j];
        const float vj = v[j];
        acc.x += kj * kj;
        acc.y += qj * qj;
        acc.z += (kj * gks[j]) * (qj * gqs[j]);
        acc.w += vj * vj;
    }

    __shared__ float4 red[THREADS / 32];
    const float4 tot = block_sum4(acc, red);
    const float inv_d = 1.0f / (float)d;
    const float rk = rsqrtf(tot.x * inv_d + eps);
    const float rq = rsqrtf(tot.y * inv_d + eps);
    const float dot = tot.z * rk * rq * inv_sqrt_d;
    const float mag = sqrtf(fminf(fmaxf(fabsf(dot), 1e-6f), 1e30f));
    const float sgn = (dot > 0.f ? 1.f : 0.f) - (dot < 0.f ? 1.f : 0.f);
    const float gate = fast_exp::sigmoid<float>(sgn * mag);
    const float rg = rsqrtf(gate * gate * tot.w * inv_d + eps);

    float4* qo4 = reinterpret_cast<float4*>(q);
    float4* o4 = reinterpret_cast<float4*>(out);
    const float4* gc4 = reinterpret_cast<const float4*>(gcs);
    auto store = [&](int i, float4 qq, float4 vv) {
        const float4 g = scale4(vv, gate);
        qo4[i] = add4(qq, g);
        o4[i] = mul4(scale4(g, rg), gc4[i]);
    };
    #pragma unroll
    for (int c = 0; c < CACHE; ++c) {
        const int i = (int)threadIdx.x + c * THREADS;
        if (i < vec) store(i, held_q[c], held_v[c]);
    }
    for (int i = (int)threadIdx.x + CACHE * THREADS; i < vec; i += THREADS) {
        store(i, q4[i], v4[i]);
    }
    for (int j = (vec << 2) + (int)threadIdx.x; j < d; j += THREADS) {
        const float g = v[j] * gate;
        q[j] = q[j] + g;
        out[j] = g * rg * gcs[j];
    }
}

// ── ple_conv ───────────────────────────────────────────────────────────────
// Rows on x; `THREADS`-wide chunks of channels on y — `float4` channels on the
// vector path. Tap k reads `(kern − 1 − k)·dil` rows back in the row's own
// sequence: a row of this wave when that is at or past the segment's start,
// otherwise a row of the sequence's history. The taps sum in order k = 0, 1, …
// and the SiLU is `acc · sigmoid(acc)`, as the eager chain computes them; the
// result is added into the residual.
//
//   wt  `[kern, hc·d]` — the checkpoint's `[hc·d, kern]` weight transposed once
//       at load, so a tap's weights for consecutive channels are consecutive
//
// The row's span is the same for every thread of the block, so the search is
// uniform: no divergence, and a wave has few spans to walk.
extern "C" __global__ void __launch_bounds__(THREADS, CONV_MIN_BLOCKS) ple_conv_kernel(
    const float* __restrict__ normalized,  // [n, hc·d]
    const float* __restrict__ wt,          // [kern, hc·d]
    const SpanDesc* __restrict__ spans,
    float* __restrict__ res,               // [n, hc·d], read and written
    int n_spans,
    int hcd,
    int kern,
    int dil,
    int hist_rows,
    int vec_ok
) {
    const int row = (int)blockIdx.x;
    int si = 0;
    while (si + 1 < n_spans && spans[si + 1].start <= row) ++si;
    const SpanDesc sp = spans[si];
    const int p = row - (int)sp.start;
    const int vec = vec_width(hcd, vec_ok);
    const int col = (int)blockIdx.y * THREADS + (int)threadIdx.x;

    if (vec > 0) {
        if (col >= vec) return;
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int k = 0; k < kern; ++k) {
            const int back = (kern - 1 - k) * dil;
            const float4 src = p >= back
                ? reinterpret_cast<const float4*>(normalized + (long long)(row - back) * hcd)[col]
                : reinterpret_cast<const float4*>(sp.hist + (long long)(hist_rows + p - back) * hcd)[col];
            const float4 term = mul4(src, reinterpret_cast<const float4*>(wt + (long long)k * hcd)[col]);
            acc = k == 0 ? term : add4(acc, term);
        }
        const float4 sg = fast_exp::sigmoid4<float>(acc);
        float4* r = reinterpret_cast<float4*>(res + (long long)row * hcd) + col;
        *r = add4(*r, mul4(acc, sg));
    } else {
        if (col >= hcd) return;
        float acc = 0.f;
        for (int k = 0; k < kern; ++k) {
            const int back = (kern - 1 - k) * dil;
            const float src = p >= back
                ? normalized[(long long)(row - back) * hcd + col]
                : sp.hist[(long long)(hist_rows + p - back) * hcd + col];
            const float term = src * wt[(long long)k * hcd + col];
            acc = k == 0 ? term : acc + term;
        }
        float* r = res + (long long)row * hcd + col;
        *r = *r + acc * fast_exp::sigmoid<float>(acc);
    }
}

// ── ple_history ────────────────────────────────────────────────────────────
// Each sequence's next history: the last `hist_rows` rows of
// `[history ; segment]`, into its spare buffer. One thread per element — a
// `float4` on the vector path — over every (span, row, column), so the grid
// covers the card: a few spans of nine rows is too little work to give each
// row a block of its own, which left all but a handful of SMs idle.
extern "C" __global__ void __launch_bounds__(THREADS) ple_history_kernel(
    const float* __restrict__ normalized,  // [n, hc·d]
    const SpanDesc* __restrict__ spans,
    int n_spans,
    int hcd,
    int hist_rows,
    int vec_ok
) {
    const int vec = vec_width(hcd, vec_ok);
    const long long cols = vec > 0 ? vec : hcd;
    const long long t = (long long)blockIdx.x * THREADS + threadIdx.x;
    const long long per_span = (long long)hist_rows * cols;
    if (t >= per_span * n_spans) return;
    const int si = (int)(t / per_span);
    const long long rem = t - (long long)si * per_span;
    const int i = (int)(rem / cols);
    const long long c = rem - (long long)i * cols;
    const SpanDesc sp = spans[si];
    const long long j = sp.len + i;  // row j of [history ; segment]
    const float* src = j < hist_rows
        ? sp.hist + j * hcd
        : normalized + (sp.start + j - hist_rows) * hcd;
    float* dst = sp.new_hist + (long long)i * hcd;
    if (vec > 0) {
        reinterpret_cast<float4*>(dst)[c] = reinterpret_cast<const float4*>(src)[c];
    } else {
        dst[c] = src[c];
    }
}

} // namespace ple_fused

extern "C" void run_ple_gate(
    const float* kv, float* res, const float* gk, const float* gq, const float* gc,
    float* normalized, int32_t n, int32_t hc, int32_t d, int32_t kv_stride,
    float eps, float inv_sqrt_d, int32_t vec_ok, void* stream
) {
    if (n <= 0 || hc <= 0 || d <= 0) return;
    ple_fused::ple_gate_kernel<<<(unsigned)(n * hc), ple_fused::THREADS, 0, (cudaStream_t)stream>>>(
        kv, res, gk, gq, gc, normalized, hc, d, kv_stride, eps, inv_sqrt_d, vec_ok);
}

// `spans` is a device array of `n_spans` four-word descriptors (start, len,
// hist, new_hist), tiling rows `0..n` in order. The history pointers must be
// 16-byte aligned whenever `vec_ok` is set — the host's check, with the rest.
extern "C" void run_ple_conv(
    const float* normalized, const float* wt, const void* spans, float* res,
    int32_t n, int32_t n_spans, int32_t hcd, int32_t kern, int32_t dil, int32_t hist_rows,
    int32_t vec_ok, void* stream
) {
    if (n <= 0 || n_spans <= 0 || hcd <= 0 || kern <= 0) return;
    const int vec = ple_fused::vec_width(hcd, vec_ok);
    const int cols = vec > 0 ? vec : hcd;
    const dim3 grid((unsigned)n, (unsigned)((cols + ple_fused::THREADS - 1) / ple_fused::THREADS), 1);
    const auto* sp = reinterpret_cast<const ple_fused::SpanDesc*>(spans);
    ple_fused::ple_conv_kernel<<<grid, ple_fused::THREADS, 0, (cudaStream_t)stream>>>(
        normalized, wt, sp, res, n_spans, hcd, kern, dil, hist_rows, vec_ok);
    if (hist_rows > 0) {
        const long long elems = (long long)n_spans * hist_rows * cols;
        const unsigned blocks = (unsigned)((elems + ple_fused::THREADS - 1) / ple_fused::THREADS);
        ple_fused::ple_history_kernel<<<blocks, ple_fused::THREADS, 0, (cudaStream_t)stream>>>(
            normalized, sp, n_spans, hcd, hist_rows, vec_ok);
    }
}
