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
// The one q8a128 tile emitter, for the epilogue that hands the out-projection
// its int8 operand.
#include "../quantize/q8a128_tile.cuh"

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

// The conv epilogue's activation.
__device__ __forceinline__ float dn_silu(float x) {
    return x * dn_sigmoid(x);
}

// One l2-normed Q|K element: `sv / max(sqrt(Σx²), eps)` — the floor on the
// ROOT, as `l2_norm` in mix.rs takes it.
__device__ __forceinline__ float dn_l2_scale(float sv, float sumsq, float eps) {
    return sv / fmaxf(sqrtf(sumsq), eps);
}

// The per-token log-decay gate g = a · softplus(α + dt_bias), ≤ 0 since a < 0.
__device__ __forceinline__ float dn_decay_gate(float a_neg, float alpha, float dt_bias) {
    return a_neg * dn_softplus(alpha + dt_bias);
}

// SiLU then, for the Q|K columns, the per-head l2 norm — the epilogue both
// conv kernels apply so their output IS the mixer's operand buffer.
//
// The short-span kernel (delta_net_short_span_kernel.cuh) computes the same
// element from the same helpers and the same 128-wide reduction tree, so the
// two produce the same bits.
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
    const float sv = dn_silu(acc);
    if (c >= qk_channels) return sv;
    red[tid] = sv * sv;
    __syncthreads();
    const int base = tid & ~(DN_HEAD_DIM - 1); // this thread's head group
    for (int off = DN_HEAD_DIM / 2; off >= 1; off >>= 1) {
        if ((tid & (DN_HEAD_DIM - 1)) < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    return dn_l2_scale(sv, red[base], eps);
}

namespace delta_net {

// ============================================================================
// Row-wise epilogue over the whole wave, shared by the prefill scan and the
// decode step: per (token, V head),
//   gated = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ zgate(z)
// — `rms_norm_per_head` and the z-gate in one pass instead of ~6 launches and
// three full-width intermediates. One block per row; d is a runtime width
// (≤ 256), threads stripe it and reduce the mean in shared memory.
//
// The z-gate is a TEMPLATE parameter, not a runtime branch: the Qwen3.5
// lineage gates with SiLU(z), qwen4exp with sigmoid(z) — the one numerical
// difference between the two generations' GDN — and baking the choice per
// instantiation keeps the epilogue branch-free for both.
//
// With `q8` set (and d = 128, one q8a128 tile per row) the block also writes
// its row as tile `blockIdx.x` of the q8a1024 operand the out-projection's int8
// GEMM reads — row-major tiles of the flat [T, h_v·d] output, the layout a
// standalone quantize of `out` writes, from the same floats and through the
// same tile arithmetic, so the bytes are that quantize's. One launch of the
// standalone quantize per DeltaNet layer goes with it. An int8 projection
// reads the operand alone, so `out` may then be null and the F32 store — a
// full-width write nothing reads — is skipped.
// ============================================================================
template <bool SIGMOID_GATE>
static __global__ void delta_net_norm_gate_f32_kernel(
        const float* __restrict__ o,     // [T, h_v·d]
        const float* __restrict__ z,     // [T, h_v·d] raw (pre-SiLU)
        const float* __restrict__ gain,  // [d]
        float*       __restrict__ out,   // [T, h_v·d], or null when `q8` is set
        int d,
        float eps,
        uint8_t*     __restrict__ q8,    // q8a1024 operand of `out`, or null
        int sum_norm) {
    __shared__ float warp_sums[8];
    __shared__ float row_vals[DN_HEAD_DIM];
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
        const float gate = SIGMOID_GATE ? dn_sigmoid(zv) : zv * dn_sigmoid(zv);
        const float v = o[row + x] * inv * gain[x] * gate;
        if (out != nullptr) out[row + x] = v;
        if (q8 != nullptr) row_vals[x] = v;
    }
    // Block-uniform: the launcher admits `q8` only at d = DN_HEAD_DIM, where
    // thread x stored element x above and warp 0's lane l now takes [4l, 4l+4)
    // — the tile emitter's lane mapping.
    if (q8 != nullptr) {
        __syncthreads();
        if (tid < 32) {
            emit_q8a128_tile(q8, (int)blockIdx.x, tid,
                             row_vals[4 * tid], row_vals[4 * tid + 1],
                             row_vals[4 * tid + 2], row_vals[4 * tid + 3], sum_norm);
        }
    }
}

// 0 when launched, 1 when the shape is refused (nothing written): `d` outside
// 1..=256, an operand requested at a width other than one tile per row, or
// neither an output nor an operand to write.
static inline int launch_norm_gate_f32(
        const float* o,
        const float* z,
        const float* gain,
        float* out,
        int rows,
        int d,
        float eps,
        int sigmoid_gate,
        uint8_t* q8,
        int sum_norm,
        cudaStream_t stream) {
    if (rows <= 0 || d <= 0 || d > 256) return 1;
    if (q8 != nullptr && d != DN_HEAD_DIM) return 1;
    if (out == nullptr && q8 == nullptr) return 1;
    if (sigmoid_gate != 0) {
        delta_net_norm_gate_f32_kernel<true><<<rows, 128, 0, stream>>>(
            o, z, gain, out, d, eps, q8, sum_norm);
    } else {
        delta_net_norm_gate_f32_kernel<false><<<rows, 128, 0, stream>>>(
            o, z, gain, out, d, eps, q8, sum_norm);
    }
    return 0;
}

} // namespace delta_net
