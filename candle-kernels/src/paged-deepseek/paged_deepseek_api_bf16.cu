// =============================================================================
// paged_deepseek_api_bf16.cu — BF16 entry for the DeepSeek hybrid decode
// (single translation unit for the fork's kernels + probes).
//
// DeepSeek-V4-Flash geometry: HEAD_DIM=512 (single latent, K≡V, MQA),
// ROPE_DIM=64 (nope‖rope split). One instantiation — the model has exactly one
// attention geometry, and each extra instantiation is real compile time.
// =============================================================================

#include "deepseek_decode_kernel.cuh"

#include <cuda_bf16.h>

extern "C" void run_paged_deepseek_prefill_bf16(
    const void* q_ptr,          // [total_q, H, 512] bf16, pre-RoPE
    const uint8_t* headers_ptr, // SlotHeader[1] (arena holds the committed prefix)
    void* o_ptr,                // [total_q, H, 512] bf16, de-rotated
    const uint32_t* q_pos,      // [total_q]
    const void* kv_fresh,       // [fresh_rows, 512] bf16 pre-RoPE (this layer's latents)
    const float* comp_ptr,
    const uint32_t* comp_pos,
    const uint32_t* comp_idx, // [total_q, max_sel]
    const uint32_t* comp_cnt, // [total_q]
    const float* sinks,
    const float* rope_freqs,
    int32_t total_q,
    int32_t n_q_head,
    float softmax_scale,
    int32_t window_size,
    int32_t max_sel,
    int32_t fresh_rows,
    int32_t fresh_base,
    int32_t num_splits_override,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    deepseek_attn::launch_deepseek_prefill<__nv_bfloat16, 512, 64>(
        (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr, q_pos,
        (const __nv_bfloat16*)kv_fresh, comp_ptr, comp_pos, comp_idx, comp_cnt,
        sinks, rope_freqs, total_q, n_q_head, softmax_scale, window_size,
        max_sel, fresh_rows, fresh_base, num_splits_override, stream);
}

extern "C" void run_paged_deepseek_glue_scatter_bf16(
    const void* kv,             // [rows, 512] bf16 pre-RoPE latents
    const uint8_t* headers_ptr, // SlotHeader[1]
    const uint32_t* slices,     // [rows] gap block index
    const uint32_t* in_blk,     // [rows] in-block offset
    int32_t rows,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (rows <= 0) return;
    int threads = 256;
    int blocks = (rows * 32 + threads - 1) / threads;
    deepseek_attn::deepseek_glue_scatter_kernel<__nv_bfloat16, 512>
        <<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)kv, headers_ptr, slices, in_blk, rows);
}

// Regression probe: evaluates the kernel-side `ds_exp` on a device array so
// the CPU mirror's replica can be asserted bit-identical over a full sweep.
__global__ void deepseek_exp_probe_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = deepseek_attn::ds_exp(in[i]);
}

extern "C" void run_deepseek_exp_probe(
    const float* in, float* out, int32_t n, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    deepseek_exp_probe_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

// Regression probe: the kernel-side RoPE trig (`rope_angle` + `ds_sincos`) for
// (pos, freq) pairs → interleaved (sin, cos) — asserted bit-identical to the
// CPU mirror.
__global__ void deepseek_sincos_probe_kernel(
    const int* pos, const float* freq, float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r, s, c;
        int k;
        deepseek_attn::rope_angle(pos[i], freq[i], r, k);
        deepseek_attn::ds_sincos(r, k, s, c);
        out[2 * i] = s;
        out[2 * i + 1] = c;
    }
}

extern "C" void run_deepseek_sincos_probe(
    const int* pos, const float* freq, float* out, int32_t n, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    deepseek_sincos_probe_kernel<<<blocks, threads, 0, stream>>>(pos, freq, out, n);
}

extern "C" void run_paged_deepseek_decode_bf16(
    const void* q_ptr,          // [slots, H, 512] bf16, pre-RoPE
    const uint8_t* headers_ptr, // SlotHeader[slots]
    void* o_ptr,                // [slots, H, 512] bf16, de-rotated final output
    const void* kv_new,         // [slots, 512] bf16, pre-RoPE latent
    const float* comp_ptr,      // [G_total, 512] f32, pre-RoPE compressed entries
    const uint32_t* comp_pos,   // [G_total] group-start positions
    const uint32_t* comp_idx,   // [slots, max_sel] ascending GIDs (0xFFFFFFFF pad)
    const uint32_t* comp_cnt,   // [slots] selected count
    const float* sinks,         // [H] per-head sink logits
    const float* rope_freqs,    // [32] YaRN-adjusted inverse frequencies
    int32_t num_slots,
    int32_t n_q_head,
    float softmax_scale,
    int32_t window_size,
    int32_t max_sel,
    int32_t num_splits_override,
    int32_t commit_write_len, // 0 = skip the on-device write-len advance (wave)
    float* dbg,  // nullable stage-dump (mirror-oracle diagnostics)
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    deepseek_attn::launch_deepseek_decode<__nv_bfloat16, 512, 64>(
        (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr,
        (const __nv_bfloat16*)kv_new, comp_ptr, comp_pos, comp_idx, comp_cnt,
        sinks, rope_freqs, num_slots, n_q_head, softmax_scale, window_size,
        max_sel, num_splits_override, commit_write_len != 0, stream, dbg);
}
