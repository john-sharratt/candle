// =============================================================================
// image_rope_rows — turn listed rows of Q or K in place by their own (cos, sin)
// =============================================================================
//
// An image row's multimodal position is its slot plus an offset per axis that
// depends only on the row's place in its image, and RoPE composes, so each
// image row's Q and K are turned once by that offset before any attention
// kernel rotates them by the slot (`docs/flash_next_vision.md` §4.4). This is
// the turn: for each listed row `rows[i]` of `x` (`[n_rows, heads, dim]`,
// dense), every head's rotary pair `f < pairs` — channels `(f, f + dim/2)` in
// the kernels' slot order — is rotated by `(cos, sin)[i, f]`, NeoX style:
//
//   lo' = lo·c − hi·s,   hi' = hi·c + lo·s
//
// Every other channel is left as it is: the pairs past `pairs` do not rotate
// at all, and the pass-through channels are not touched. In F32 — each
// product and the sum rounded as written, no FMA — the result rounded once to
// `x`'s type.
//
// In place: each pair is read and written by one thread, and `rows` names a
// row at most once, so nothing a thread reads is written by another.
//
// GRID AND BLOCK
// --------------
//   Grid:  (n_listed, 1, 1) — one block per listed row
//   Block: (THREADS, 1, 1)  — the row's `heads · pairs` pairs, strided
//
// Consecutive threads take consecutive pairs of a head, so a warp reads and
// writes two runs of 32 consecutive elements — the low and high halves.
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace image_rope_rows {

constexpr int THREADS = 256;

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }

template <typename T>
__global__ void __launch_bounds__(THREADS) kernel(
    T *__restrict__ x,
    const uint32_t *__restrict__ rows,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    int heads,
    int dim,
    int pairs) {
  const int i = blockIdx.x;
  T *row = x + (long long)__ldg(rows + i) * heads * dim;
  const float *c_row = cos + (long long)i * pairs;
  const float *s_row = sin + (long long)i * pairs;
  const int half = dim / 2;
  for (int t = threadIdx.x; t < heads * pairs; t += blockDim.x) {
    const int h = t / pairs;
    const int f = t - h * pairs;
    T *p = row + (long long)h * dim;
    const float c = __ldg(c_row + f);
    const float s = __ldg(s_row + f);
    const float lo = to_f(p[f]);
    const float hi = to_f(p[f + half]);
    // Each product rounded before the sum, never contracted into an FMA, so
    // the result is the host's `lo·c − hi·s` in f32 to the bit.
    p[f] = from_f<T>(__fsub_rn(__fmul_rn(lo, c), __fmul_rn(hi, s)));
    p[f + half] = from_f<T>(__fadd_rn(__fmul_rn(hi, c), __fmul_rn(lo, s)));
  }
}

template <typename T>
static void launch(void *x, const uint32_t *rows, const float *cos, const float *sin,
                   int n_listed, int heads, int dim, int pairs, cudaStream_t stream) {
  kernel<T><<<n_listed, THREADS, 0, stream>>>((T *)x, rows, cos, sin, heads, dim, pairs);
}

} // namespace image_rope_rows

// `dtype`: 0 F32, 1 F16, 2 BF16. Returns 0 on launch (or nothing to do), -1 for
// a type with no instantiation, -2 for a shape the turn cannot apply to: `dim`
// odd, or more pairs than half the head. The caller validates `rows` against
// `x`'s row count; the kernel trusts it.
extern "C" int32_t run_image_rope_rows(
    int32_t dtype,
    void *x,
    const uint32_t *rows,
    const float *cos,
    const float *sin,
    int32_t n_listed,
    int32_t heads,
    int32_t dim,
    int32_t pairs,
    void *stream) {
  if (n_listed <= 0 || heads <= 0 || pairs <= 0) {
    return 0;
  }
  if (dim % 2 != 0 || pairs > dim / 2) {
    return -2;
  }
  cudaStream_t s = (cudaStream_t)stream;
  switch (dtype) {
  case 0: image_rope_rows::launch<float>(x, rows, cos, sin, n_listed, heads, dim, pairs, s); return 0;
  case 1: image_rope_rows::launch<__half>(x, rows, cos, sin, n_listed, heads, dim, pairs, s); return 0;
  case 2: image_rope_rows::launch<__nv_bfloat16>(x, rows, cos, sin, n_listed, heads, dim, pairs, s); return 0;
  default: return -1;
  }
}
