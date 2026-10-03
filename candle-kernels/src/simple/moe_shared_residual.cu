// =============================================================================
// MoE SHARED-EXPERT RESIDUAL — the layer's FFN output folded into the residual
// =============================================================================
//
//   x[t, j] += narrow(routed[t, j] + shared[t, j] · sigmoid(gate[t]))
//
// A MoE layer with a gated shared expert (the Qwen3.5 lineage) ends in four
// elementwise launches: the gate's sigmoid, its broadcast multiply into the
// shared expert's output, the add of the routed experts' sum, and the residual
// add. Each round-trips an `[n, d]` buffer and none does enough arithmetic to
// cover its launch. This is all four in one pass that reads the three parts and
// updates the residual in place.
//
// **The same arithmetic as those four, so the same bits.** Every step rounds to
// the type its launch stored: the sigmoid is `usigmoid`'s own (`sigmoid_fwd`),
// the product is rounded before the sum (`*_rn` — a contracted FMA would not
// be), and the narrowing to the residual's type happens after the sum, where the
// eager chain's `to_dtype` did. The parts are in the FFN's working type and the
// residual in the stream's; they differ only when the stream is F16, whose SwiGLU
// runs in BF16.
//
// `gate` is one scalar per row, `gate_stride` elements apart: the first column of
// the gate projection's KO-padded output, read in place rather than compacted.
//
// In place, like `Tensor::add_mut`: each element is read and written by one
// thread at one index, so `x` aliases nothing it reads.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "sigmoid_fwd.cuh"

namespace moe_shared_residual {

constexpr int THREADS = 256;

// One rounded step per eager launch.
__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ __nv_bfloat16 mul_rn(__nv_bfloat16 a, __nv_bfloat16 b) { return __hmul_rn(a, b); }
__device__ __forceinline__ __nv_bfloat16 add_rn(__nv_bfloat16 a, __nv_bfloat16 b) { return __hadd_rn(a, b); }
__device__ __forceinline__ __half add_rn(__half a, __half b) { return __hadd_rn(a, b); }

// The FFN output's narrowing into the residual's type — `to_dtype`'s conversion.
template<typename X, typename P> __device__ __forceinline__ X narrow(P v);
template<> __device__ __forceinline__ float narrow<float, float>(float v) { return v; }
template<> __device__ __forceinline__ __nv_bfloat16 narrow<__nv_bfloat16, __nv_bfloat16>(__nv_bfloat16 v) { return v; }
template<> __device__ __forceinline__ __half narrow<__half, __nv_bfloat16>(__nv_bfloat16 v) {
    return __float2half(__bfloat162float(v));
}

template<typename P, typename X>
__global__ void __launch_bounds__(THREADS) kernel(
    X* x,                             // [n, d], read and written
    const P* __restrict__ routed,     // [n, d]
    const P* __restrict__ shared,     // [n, d]
    const P* __restrict__ gate,       // [n], row stride `gate_stride`
    int d,
    int gate_stride
) {
    const int row = (int)blockIdx.x;
    const int col = (int)blockIdx.y * THREADS + (int)threadIdx.x;
    if (col >= d) return;
    const P g = sigmoid_fwd<P>(gate[(long long)row * gate_stride]);
    const long long i = (long long)row * d + col;
    const P h = add_rn(routed[i], mul_rn(shared[i], g));
    x[i] = add_rn(x[i], narrow<X, P>(h));
}

} // namespace moe_shared_residual

// Mirrors `MOE_SHARED_RESIDUAL_LAUNCHED` / `_REFUSED` in `moe_shared_residual.rs`.
#define MOE_SHARED_RESIDUAL_LAUNCHED 0
#define MOE_SHARED_RESIDUAL_REFUSED 1

// dtype codes are `MoeScatterDType`'s: 0 = f32, 1 = f16, 2 = bf16. The pairs
// instantiated are the ones a layer can produce — parts and residual alike, or
// BF16 parts into an F16 stream — and any other pair is refused, never launched
// with a guess.
extern "C" int32_t run_moe_shared_residual(
    int32_t parts_dtype, int32_t x_dtype,
    void* x, const void* routed, const void* shared, const void* gate,
    int32_t n, int32_t d, int32_t gate_stride, void* stream
) {
    if (n < 0 || d <= 0 || gate_stride < 1) return MOE_SHARED_RESIDUAL_REFUSED;
    const dim3 grid((unsigned)n, (unsigned)((d + moe_shared_residual::THREADS - 1) / moe_shared_residual::THREADS));
    const cudaStream_t s = (cudaStream_t)stream;
#define MSR_LAUNCH(P, X)                                                                     \
    if (n > 0) {                                                                             \
        moe_shared_residual::kernel<P, X><<<grid, moe_shared_residual::THREADS, 0, s>>>(     \
            (X*)x, (const P*)routed, (const P*)shared, (const P*)gate, d, gate_stride);      \
    }                                                                                        \
    return MOE_SHARED_RESIDUAL_LAUNCHED
    if (parts_dtype == 0 && x_dtype == 0) { MSR_LAUNCH(float, float); }
    if (parts_dtype == 2 && x_dtype == 2) { MSR_LAUNCH(__nv_bfloat16, __nv_bfloat16); }
    if (parts_dtype == 2 && x_dtype == 1) { MSR_LAUNCH(__nv_bfloat16, __half); }
#undef MSR_LAUNCH
    return MOE_SHARED_RESIDUAL_REFUSED;
}
