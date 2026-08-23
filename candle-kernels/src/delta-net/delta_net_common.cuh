#pragma once
// Device helpers shared by the Gated DeltaNet kernels (decode step, conv
// step, and the fused prefill scan), plus the norm/SiLU-gate epilogue kernel
// both phases end with. The reference for every formula here is
// candle-transformers/src/models/delta_net/mix.rs — these must match it
// exactly, because the tensor-op fallback path computes the same values with
// the candle ops and the parity tests compare the two.
//
// The epilogue kernel is concrete (non-template) and `static`: this header is
// compiled by the single translation unit delta_net_api_f32.cu, and internal
// linkage keeps a second includer from colliding at link time.

#include <cuda_runtime.h>
#include <math.h>

// DeltaNet head width the fused kernels are compiled for (d_k == d_v), and
// the width of one l2-norm group in the Q|K stack.
#define DN_HEAD_DIM 128

// softplus(x) = max(x, 0) + ln(1 + e^{-|x|})  (the numerically stable form
// `softplus` in delta_net.rs uses).
__device__ __forceinline__ float dn_softplus(float x) {
    return fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
}

__device__ __forceinline__ float dn_sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

// SiLU then, for the Q|K columns, the per-head l2 norm — the epilogue both
// conv kernels apply so their output IS the mixer's operand buffer.
//
// The reference is `l2_norm` exactly: `x / max(sqrt(Σx²), eps)`, the floor on
// the ROOT, over each 128-dim head row of the SiLU'd values. The reduction is
// block-local: `qk_channels = 2·h_k·128 = h_k·256`, so a 256-thread block
// whose channel base is a multiple of 256 holds either two complete head
// groups or none of the Q|K region — never a fragment. V-region calls return
// without touching `red` or syncing, so partial trailing blocks (which are
// always V) cannot deadlock the reduction.
//
// `red` is the caller's 256-float smem scratch. Returns the value to store.
__device__ __forceinline__ float dn_silu_norm_epilogue(
        float acc, int c, int qk_channels, float eps, int tid, float* red) {
    const float sv = acc * dn_sigmoid(acc);
    if (c >= qk_channels) return sv;
    red[tid] = sv * sv;
    __syncthreads();
    const int base = tid & ~(DN_HEAD_DIM - 1); // this thread's head group
    for (int off = DN_HEAD_DIM / 2; off >= 1; off >>= 1) {
        if ((tid & (DN_HEAD_DIM - 1)) < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    return sv / fmaxf(sqrtf(red[base]), eps);
}

namespace delta_net {

// ============================================================================
// Row-wise epilogue over the whole wave, shared by the prefill scan and the
// decode step: per (token, V head),
//   gated = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ SiLU(z)
// — `rms_norm_per_head` and the z-gate in one pass instead of ~6 launches and
// three full-width intermediates. One block per row; d is a runtime width
// (≤ 256), threads stripe it and reduce the mean in shared memory.
// ============================================================================
static __global__ void delta_net_norm_gate_f32_kernel(
        const float* __restrict__ o,     // [T, h_v·d]
        const float* __restrict__ z,     // [T, h_v·d] raw (pre-SiLU)
        const float* __restrict__ gain,  // [d]
        float*       __restrict__ out,   // [T, h_v·d]
        int d,
        float eps) {
    __shared__ float warp_sums[8];
    const size_t row = (size_t)blockIdx.x * d;
    const int tid = (int)threadIdx.x;

    float ss = 0.f;
    for (int x = tid; x < d; x += (int)blockDim.x) {
        const float ov = o[row + x];
        ss += ov * ov;
    }
    ss += __shfl_down_sync(0xffffffffu, ss, 16);
    ss += __shfl_down_sync(0xffffffffu, ss, 8);
    ss += __shfl_down_sync(0xffffffffu, ss, 4);
    ss += __shfl_down_sync(0xffffffffu, ss, 2);
    ss += __shfl_down_sync(0xffffffffu, ss, 1);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.f;
        for (int i = 0; i < ((int)blockDim.x + 31) / 32; ++i) tot += warp_sums[i];
        warp_sums[0] = rsqrtf(tot / (float)d + eps);
    }
    __syncthreads();
    const float inv = warp_sums[0];

    for (int x = tid; x < d; x += (int)blockDim.x) {
        const float zv = z[row + x];
        out[row + x] = o[row + x] * inv * gain[x] * (zv * dn_sigmoid(zv));
    }
}

static inline void launch_norm_gate_f32(
        const float* o,
        const float* z,
        const float* gain,
        float* out,
        int rows,
        int d,
        float eps,
        cudaStream_t stream) {
    if (rows <= 0 || d <= 0 || d > 256) return;
    delta_net_norm_gate_f32_kernel<<<rows, 128, 0, stream>>>(
        o, z, gain, out, d, eps);
}

} // namespace delta_net
