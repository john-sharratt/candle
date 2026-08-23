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
// One block per V head; threads stripe the d_v state rows. k and q are staged
// in shared memory once per block. d_k and d_v are runtime arguments bounded
// by DELTA_NET_MAX_HEAD_DIM (shared-memory budget: 2 * 256 * 4 B = 2 KB).
//
// Concrete (non-template) kernels: this header is compiled by the single
// translation unit delta_net_api_f32.cu; `static` keeps the definitions
// TU-local so a second includer cannot collide at link time.

#include "delta_net_common.cuh"

#define DELTA_NET_MAX_HEAD_DIM 256

namespace delta_net {

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

    __shared__ float sh_k[DELTA_NET_MAX_HEAD_DIM];
    __shared__ float sh_q[DELTA_NET_MAX_HEAD_DIM];

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
    // twice): each `j` is read before it is written, so in-place stays correct.
    for (int i = threadIdx.x; i < d_v; i += blockDim.x) {
        const float* srow_in = state_in + ((size_t)h * d_v + i) * d_k;
        float* srow_out = state_out + ((size_t)h * d_v + i) * d_k;
        // Decayed prediction the state makes for k.
        float pred = 0.f;
        for (int j = 0; j < d_k; ++j) {
            pred += srow_in[j] * decay * sh_k[j];
        }
        const float err = b * (v[(size_t)h * d_v + i] - pred);
        // Update the row and read the output with the post-update state.
        float out = 0.f;
        for (int j = 0; j < d_k; ++j) {
            const float s_new = srow_in[j] * decay + err * sh_k[j];
            srow_out[j] = s_new;
            out += s_new * sh_q[j];
        }
        orow[(size_t)h * d_v + i] = out;
    }
}

// Causal depthwise conv, one token step per decode sequence, batched over
// the wave: grid ((channels+255)/256, n_decode). Tails are carried in place
// (each sequence's tail lives in its own allocation — the pointer table
// again), and the SiLU + Q|K-norm epilogue makes the output row the same
// operand-buffer contract as the prefill conv.
//   x      : [T_wave, channels]      the wave's pre-conv fused QKV rows
//   kernel : [channels, kwidth]
//   tails  : [n_decode] device f32* to [channels, kwidth-1] (shift left,
//            append the RAW x — the conv window wants pre-activation values)
//   rows   : [n_decode] each sequence's row in x/y
//   y      : [T_wave, channels]      y = epilogue(sum_j kern[c,j]*window[j])
// window = [tail | x] so the output sees inputs t-K+1 ..= t.
static __global__ void delta_net_conv_decode_f32_kernel(
        const float*        __restrict__ x,
        const float*        __restrict__ kernel,
        const long long*    __restrict__ tails,
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
    float* trow = ((float*)tails[seq]) + (size_t)c * tcols;
    const float xv = x[(size_t)row * channels + c];
    const float* krow = kernel + (size_t)c * kwidth;

    float acc = krow[kwidth - 1] * xv;
    for (int j = 0; j < tcols; ++j) {
        acc += krow[j] * trow[j];
    }
    y[(size_t)row * channels + c] =
        dn_silu_norm_epilogue(acc, c, qk_channels, eps, (int)threadIdx.x, red);

    // Shift the tail left and append this token.
    for (int j = 0; j + 1 < tcols; ++j) {
        trow[j] = trow[j + 1];
    }
    if (tcols > 0) {
        trow[tcols - 1] = xv;
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
    const int threads = d_v < 128 ? ((d_v + 31) / 32) * 32 : 128;
    dim3 grid(n_v_heads, n_decode);
    delta_net_decode_step_f32_kernel<<<grid, threads, 0, stream>>>(
        states, states_out, conved, rows, alpha, beta_lin, dt_bias, a_neg, o,
        d_k, d_v, n_v_heads, n_k_heads, tok_stride, q_scale);
}

static inline void launch_conv_decode_f32(
        const float* x,
        const float* kernel,
        const long long* tails,
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
        x, kernel, tails, rows, y, channels, kwidth, qk_channels, eps);
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
