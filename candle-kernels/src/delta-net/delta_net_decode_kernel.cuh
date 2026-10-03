#pragma once
// Gated DeltaNet — decode-step and causal-conv-step kernels.
//
// The recurrence these implement is the reference in
// candle-transformers/src/models/delta_net/mix.rs (the parity oracle):
//
//   g       =  a * softplus(alpha + dt_bias)    per V head (computed here)
//   beta    =  sigmoid(beta_lin)
//   S       <- exp(g) * S                       per V head, g <= 0
//   S       <- S + beta * (v - S k) (x) k       delta-rule correction
//   o       =  S q                              post-update read
//
// All state math is F32: the state is a running sum and half precision
// drifts across a long decode (docs/qwen35_qwen38_models.md §8 risk 2).
//
// The decode step reads the mixer's own buffers through strides — no GQA
// repeat, no per-span copies, no separate q/k/v tensors (the same contract as
// the prefill scan in delta_net_prefill_kernel.cuh):
//   state : [h_v, d_v, d_k]  updated in place
//   qk    : one token's row of the l2-normed Q|K stack, [2*h_k, d_k];
//           V head h reads K head h % h_k, q scaled by q_scale on load
//   v     : one token's V columns of the conv output, [h_v, d_v]
//   alpha, beta_lin : one token's raw gate projections, [h_v]
//   dt_bias, a      : [h_v] constants
//   o     : one token's row of the wave output, [h_v, d_v]
//
// One block per (V head, sequence); warps stripe the d_v state rows and the
// lanes of a warp stripe one row. k and q are staged in shared memory once per
// block. d_k and d_v are runtime arguments bounded by DELTA_NET_MAX_HEAD_DIM
// (shared-memory budget: 2 * 256 * 4 B = 2 KB); d_k is a multiple of 4, so a
// row is whole float4s.
//
// Concrete (non-template) kernels: this header is compiled by the single
// translation unit delta_net_api_f32.cu; `static` keeps the definitions
// TU-local so a second includer cannot collide at link time.

#include "delta_net_common.cuh"

#define DELTA_NET_MAX_HEAD_DIM 256

namespace delta_net {

// float4 chunks of a state row each lane holds: a chunk is 32 lanes × 4
// floats = 128 elements of the row.
constexpr int DN_ROW_CHUNKS = DELTA_NET_MAX_HEAD_DIM / 128;

__device__ __forceinline__ float dn_warp_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
    return x;
}

// Batched over the wave's decode sequences: grid (n_v_heads, n_decode). Each
// sequence's state lives in its own allocation, so the kernel takes a device
// array of state base pointers plus each sequence's row in the wave tensors —
// the table one host upload per FORWARD builds (never per layer: a mid-sweep
// upload would serialise the launch pipeline).
static __global__ void delta_net_decode_step_f32_kernel(
        const long long*     __restrict__ states,     // [n_decode] entering f32* as i64
        const long long*     __restrict__ states_out, // [n_decode] advanced f32* as i64
        const float*         __restrict__ conved, // [T_wave, tok_stride]
        const unsigned int*  __restrict__ rows,   // [n_decode] wave rows
        const float*         __restrict__ alpha,  // [T_wave, n_v_heads] raw
        const float*         __restrict__ beta_lin,
        const float*         __restrict__ dt_bias,
        const float*         __restrict__ a_neg,
        float*               __restrict__ o,      // [T_wave, n_v_heads·d_v]
        int d_k,
        int d_v,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale) {
    const int h = blockIdx.x;   // V head
    const int seq = blockIdx.y; // decode sequence
    const int kh = h % n_k_heads;
    const int row = (int)rows[seq];
    const float* state_in = (const float*)states[seq];
    float* state_out = (float*)states_out[seq];
    const float* qk = conved + (size_t)row * tok_stride;
    const float* v = qk + ((size_t)tok_stride - (size_t)n_v_heads * d_v);
    const float* gates_a = alpha + (size_t)row * n_v_heads;
    const float* gates_b = beta_lin + (size_t)row * n_v_heads;
    float* orow = o + (size_t)row * n_v_heads * d_v;

    __shared__ __align__(16) float sh_k[DELTA_NET_MAX_HEAD_DIM];
    __shared__ __align__(16) float sh_q[DELTA_NET_MAX_HEAD_DIM];

    for (int j = threadIdx.x; j < d_k; j += blockDim.x) {
        sh_q[j] = qk[(size_t)kh * d_k + j] * q_scale;
        sh_k[j] = qk[(size_t)(n_k_heads + kh) * d_k + j];
    }
    __syncthreads();

    const float g     = a_neg[h] * dn_softplus(gates_a[h] + dt_bias[h]);
    const float decay = expf(g);
    const float b     = dn_sigmoid(gates_b[h]);

    // The entering state is read from `state_in`, the advanced state written to
    // `state_out` — the wave points the latter at the slot's OTHER buffer. Every
    // element of the row is written below, so the destination needs no
    // initialisation and carries nothing forward from whatever it last held.
    // That is what lets a failed wave roll back by simply not swapping the two
    // buffers, instead of copying the entering state aside before every wave.
    // The two may also be the same pointer (the reference path passes one buffer
    // twice): a row is read whole into registers before any of it is written,
    // so in-place stays correct.
    //
    // A warp per row: lane l holds elements 4l..4l+3 of each 128-wide chunk, so
    // the warp reads the row as one coalesced float4 sweep, keeps it in
    // registers through the prediction, the update and the output read, and
    // writes the advanced row back once — one read and one write of the state.
    // A chunk past d_k is dead for every lane at once when d_k is a multiple of
    // 128, and its registers are written on both paths.
    const int lane = (int)threadIdx.x & 31;
    const int n_warps = (int)blockDim.x >> 5;
    const float4* k4 = reinterpret_cast<const float4*>(sh_k);
    const float4* q4 = reinterpret_cast<const float4*>(sh_q);
    for (int i = (int)threadIdx.x >> 5; i < d_v; i += n_warps) {
        const float4* srow_in =
            reinterpret_cast<const float4*>(state_in + ((size_t)h * d_v + i) * d_k);
        float4* srow_out = reinterpret_cast<float4*>(state_out + ((size_t)h * d_v + i) * d_k);
        float4 s[DN_ROW_CHUNKS];
        // Decayed prediction the state makes for k.
        float pred = 0.f;
        #pragma unroll
        for (int c = 0; c < DN_ROW_CHUNKS; ++c) {
            const int j4 = c * 32 + lane;
            if (4 * j4 < d_k) {
                s[c] = srow_in[j4];
                const float4 kk = k4[j4];
                pred += s[c].x * decay * kk.x;
                pred += s[c].y * decay * kk.y;
                pred += s[c].z * decay * kk.z;
                pred += s[c].w * decay * kk.w;
            } else {
                s[c] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        pred = dn_warp_sum(pred);
        const float err = b * (v[(size_t)h * d_v + i] - pred);
        // Update the row and read the output with the post-update state.
        float out = 0.f;
        #pragma unroll
        for (int c = 0; c < DN_ROW_CHUNKS; ++c) {
            const int j4 = c * 32 + lane;
            if (4 * j4 < d_k) {
                const float4 kk = k4[j4];
                const float4 qq = q4[j4];
                float4 sn;
                sn.x = s[c].x * decay + err * kk.x;
                sn.y = s[c].y * decay + err * kk.y;
                sn.z = s[c].z * decay + err * kk.z;
                sn.w = s[c].w * decay + err * kk.w;
                srow_out[j4] = sn;
                out += sn.x * qq.x;
                out += sn.y * qq.y;
                out += sn.z * qq.z;
                out += sn.w * qq.w;
            }
        }
        out = dn_warp_sum(out);
        if (lane == 0) orow[(size_t)h * d_v + i] = out;
    }
}

// Causal depthwise conv, one token step per decode sequence, batched over
// the wave: grid ((channels+255)/256, n_decode). The entering and advanced
// tails are separate allocations named by two pointer tables — the wave writes
// the buffer it is not reading, so a failed wave leaves the entering tail
// intact and `commit_wave` installs the advance with a host pointer swap. The
// SiLU + Q|K-norm epilogue makes the output row the same operand-buffer
// contract as the prefill conv.
//   x         : [T_wave, channels]   the wave's pre-conv fused QKV rows
//   kernel    : [channels, kwidth]
//   tails     : [n_decode] device f32* to the entering [channels, kwidth-1]
//   tails_out : [n_decode] device f32* to the advanced [channels, kwidth-1]
//               (the entering tail shifted left with the RAW x appended — the
//               conv window wants pre-activation values)
//   rows      : [n_decode] each sequence's row in x/y
//   y         : [T_wave, channels]   y = epilogue(sum_j kern[c,j]*window[j])
// window = [tail | x] so the output sees inputs t-K+1 ..= t.
static __global__ void delta_net_conv_decode_f32_kernel(
        const float*        __restrict__ x,
        const float*        __restrict__ kernel,
        const long long*    __restrict__ tails,
        const long long*    __restrict__ tails_out,
        const unsigned int* __restrict__ rows,
        float*              __restrict__ y,
        int channels,
        int kwidth,
        int qk_channels,
        float eps) {
    __shared__ float red[256];
    const int seq = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int row = (int)rows[seq];

    const int tcols = kwidth - 1;
    const float* trow_in = ((const float*)tails[seq]) + (size_t)c * tcols;
    float* trow_out = ((float*)tails_out[seq]) + (size_t)c * tcols;
    const float xv = x[(size_t)row * channels + c];
    const float* krow = kernel + (size_t)c * kwidth;

    float acc = krow[kwidth - 1] * xv;
    for (int j = 0; j < tcols; ++j) {
        acc += krow[j] * trow_in[j];
    }
    y[(size_t)row * channels + c] =
        dn_silu_norm_epilogue(acc, c, qk_channels, eps, (int)threadIdx.x, red);

    // Shift the tail left and append this token.
    for (int j = 0; j + 1 < tcols; ++j) {
        trow_out[j] = trow_in[j + 1];
    }
    if (tcols > 0) {
        trow_out[tcols - 1] = xv;
    }
}

// Address arrays for cuBLAS's batched triangular solve.
//
// `cublas<t>trsmBatched` takes device arrays of per-matrix pointers and cuBLAS
// has no strided-batched trsm, so the addresses have to be materialised even
// though they are a base plus a constant stride. Building them on the host
// would be a host-to-device copy inside the chunk loop — the traffic the wave
// path exists to remove — so they are written on the device instead.
static __global__ void delta_net_batch_ptrs_kernel(
        const float*  a_base,
        long long     a_stride,
        float*        b_base,
        long long     b_stride,
        const float** a_ptrs,
        float**       b_ptrs,
        int           batch) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch) return;
    a_ptrs[i] = a_base + (long long)i * a_stride;
    b_ptrs[i] = b_base + (long long)i * b_stride;
}

static inline void launch_decode_step_f32(
        const long long* states,
        const long long* states_out,
        const float* conved,
        const unsigned int* rows,
        const float* alpha,
        const float* beta_lin,
        const float* dt_bias,
        const float* a_neg,
        float* o,
        int n_decode,
        int n_v_heads,
        int n_k_heads,
        int d_k,
        int d_v,
        int tok_stride,
        float q_scale,
        cudaStream_t stream) {
    if (n_decode <= 0 || n_v_heads <= 0 || n_k_heads <= 0 || d_k <= 0 || d_v <= 0) return;
    if (d_k > DELTA_NET_MAX_HEAD_DIM || d_v > DELTA_NET_MAX_HEAD_DIM) return;
    if (d_k % 4 != 0) return;
    const int threads = 128;
    dim3 grid(n_v_heads, n_decode);
    delta_net_decode_step_f32_kernel<<<grid, threads, 0, stream>>>(
        states, states_out, conved, rows, alpha, beta_lin, dt_bias, a_neg, o,
        d_k, d_v, n_v_heads, n_k_heads, tok_stride, q_scale);
}

static inline void launch_conv_decode_f32(
        const float* x,
        const float* kernel,
        const long long* tails,
        const long long* tails_out,
        const unsigned int* rows,
        float* y,
        int n_decode,
        int channels,
        int kwidth,
        int qk_channels,
        float eps,
        cudaStream_t stream) {
    if (n_decode <= 0 || channels <= 0 || kwidth <= 1) return;
    // Whole head groups per block — see dn_silu_norm_epilogue.
    if (qk_channels < 0 || qk_channels > channels || qk_channels % 256 != 0) return;
    const int threads = 256;
    dim3 grid((channels + threads - 1) / threads, n_decode);
    delta_net_conv_decode_f32_kernel<<<grid, threads, 0, stream>>>(
        x, kernel, tails, tails_out, rows, y, channels, kwidth, qk_channels, eps);
}

static inline void launch_batch_ptrs(
        const float*  a_base,
        long long     a_stride,
        float*        b_base,
        long long     b_stride,
        const float** a_ptrs,
        float**       b_ptrs,
        int           batch,
        cudaStream_t  stream) {
    if (batch <= 0) return;
    const int threads = 128;
    const int blocks = (batch + threads - 1) / threads;
    delta_net_batch_ptrs_kernel<<<blocks, threads, 0, stream>>>(
        a_base, a_stride, b_base, b_stride, a_ptrs, b_ptrs, batch);
}

} // namespace delta_net
