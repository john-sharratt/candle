// Gated DeltaNet — the extern "C" entry points (all F32: the recurrent state
// is an unbounded running sum, so half-precision state math is off the table
// by design — docs/qwen35_qwen38_models.md §8 risk 2).
//
// Thin by convention (see paged-decode/paged-glue/paged-latent): the kernels
// and launch logic live in the _kernel.cuh headers, this TU only binds the C
// symbols candle-kernels/src/delta-net/api.rs declares.
//
//   delta_net_decode_kernel.cuh  — decode step, conv step, TRSM address arrays
//   delta_net_prefill_kernel.cuh — fused chunked prefill scan (conv/intra/state)
//   delta_net_common.cuh         — shared device helpers + the norm/SiLU-gate
//                                  epilogue both phases end with

#include "delta_net_decode_kernel.cuh"
#include "delta_net_prefill_kernel.cuh"

extern "C" void run_delta_net_decode_step_f32(
        const long long* states,
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
        void* stream) {
    delta_net::launch_decode_step_f32(
        states, conved, rows, alpha, beta_lin, dt_bias, a_neg, o,
        n_decode, n_v_heads, n_k_heads, d_k, d_v, tok_stride, q_scale,
        (cudaStream_t)stream);
}

extern "C" void run_delta_net_conv_decode_f32(
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
        void* stream) {
    delta_net::launch_conv_decode_f32(
        x, kernel, tails, rows, y, n_decode, channels, kwidth, qk_channels,
        eps, (cudaStream_t)stream);
}

extern "C" void run_delta_net_batch_ptrs(
        const float*  a_base,
        long long     a_stride,
        float*        b_base,
        long long     b_stride,
        const float** a_ptrs,
        float**       b_ptrs,
        int           batch,
        void*         stream) {
    delta_net::launch_batch_ptrs(
        a_base, a_stride, b_base, b_stride, a_ptrs, b_ptrs, batch,
        (cudaStream_t)stream);
}

extern "C" void run_delta_net_conv_prefill_f32(
        const float* x,
        const float* kernel,
        const float* tail,
        float* y,
        float* tail_out,
        int t_len,
        int channels,
        int kwidth,
        int qk_channels,
        float eps,
        void* stream) {
    delta_net::launch_conv_prefill_f32(
        x, kernel, tail, y, tail_out, t_len, channels, kwidth, qk_channels,
        eps, (cudaStream_t)stream);
}

extern "C" void run_delta_net_prefill_intra_f32(
        const float* qk,
        const float* v,
        const float* alpha,
        const float* blin,
        const float* dt_bias,
        const float* a_neg,
        float* u,
        float* w,
        float* kq,
        float* g_cs,
        int t_len,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        void* stream) {
    delta_net::launch_prefill_intra_f32(
        qk, v, alpha, blin, dt_bias, a_neg, u, w, kq, g_cs,
        t_len, n_v_heads, n_k_heads, tok_stride, q_scale,
        (cudaStream_t)stream);
}

extern "C" void run_delta_net_prefill_state_f32(
        float* state,
        const float* qk,
        const float* u,
        const float* w,
        const float* kq,
        const float* g_cs,
        float* o,
        int t_len,
        int n_v_heads,
        int n_k_heads,
        int tok_stride,
        float q_scale,
        void* stream) {
    delta_net::launch_prefill_state_f32(
        state, qk, u, w, kq, g_cs, o, t_len, n_v_heads, n_k_heads,
        tok_stride, q_scale, (cudaStream_t)stream);
}

extern "C" void run_delta_net_norm_gate_f32(
        const float* o,
        const float* z,
        const float* gain,
        float* out,
        int rows,
        int d,
        float eps,
        void* stream) {
    delta_net::launch_norm_gate_f32(o, z, gain, out, rows, d, eps,
                                    (cudaStream_t)stream);
}
