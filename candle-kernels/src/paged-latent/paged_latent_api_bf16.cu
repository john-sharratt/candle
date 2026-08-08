// =============================================================================
// paged_latent_api_bf16.cu — BF16 entry for the paged latent-attention decode
// (single translation unit for the fork's kernels + probes).
//
// Latent geometry: HEAD_DIM=512 (single latent, K≡V, MQA), ROPE_DIM=64
// (nope‖rope split). One instantiation — there is exactly one attention
// geometry, and each extra instantiation is real compile time.
// =============================================================================

#include "latent_decode_kernel.cuh"
#include "latent_prefill_kernel.cuh"

#include <cuda_bf16.h>

extern "C" void run_paged_latent_prefill_bf16(
    const void* q_ptr,          // [total_q, H, 512] bf16, pre-RoPE
    const uint8_t* headers_ptr, // SlotHeader[1] (arena holds the committed prefix)
    void* o_ptr,                // [total_q, H, 512] bf16, de-rotated
    const uint32_t* q_pos,      // [total_q]
    const void* kv_fresh,       // [fresh_rows, 512] bf16 pre-RoPE (this layer's latents)
    const uint8_t* nope_i8,     // [G, 448] two-region cache: nope int8
    const float* nope_scale,    // [G, 14] per-nope-band scale
    const void* rope_bf,        // [G, 64] rope pre-rotation bf16
    const uint32_t* comp_pos,
    const uint32_t* comp_idx, // [total_q, max_sel]
    const uint32_t* comp_cnt, // [total_q]
    const float* sinks,
    const float* rope_tab, // factored cos/sin table (latent_common.cuh layout)
    float* partial_acc,    // caller-owned [total_q*H, num_splits, 512]
    float* partial_ml,     // caller-owned [total_q*H, num_splits, 2]
    uint8_t* comp_i8,      // per-prefill roped+quantized corpus scratch [G,512]
    float* comp_scale,     // per-band scale [G,4]
    uint8_t* comp_v8,      // per-dim-global int8 PV operand scratch [G,512]
    float* comp_vmax,      // global per-dim max|v| [512] (zeroed by caller)
    int32_t g_total,
    int32_t total_q,
    int32_t n_q_head,
    float softmax_scale,
    int32_t window_size,
    int32_t max_sel,
    int32_t fresh_rows,
    int32_t fresh_base,
    int32_t num_splits,
    int32_t store_fmt,     // writer-chunk float format tag (fresh-diagonal quant)
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    latent_attn::launch_latent_prefill<__nv_bfloat16, 512, 64>(
        (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr, q_pos,
        (const __nv_bfloat16*)kv_fresh, (const int8_t*)nope_i8, nope_scale,
        (const __nv_bfloat16*)rope_bf, comp_pos, comp_idx, comp_cnt,
        sinks, rope_tab, partial_acc, partial_ml, (int8_t*)comp_i8, comp_scale,
        (int8_t*)comp_v8, comp_vmax, g_total, total_q, n_q_head, softmax_scale,
        window_size, max_sel, fresh_rows, fresh_base, num_splits,
        store_fmt, stream);
}

extern "C" void run_paged_latent_glue_scatter_bf16(
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
    latent_attn::latent_glue_scatter_kernel<__nv_bfloat16, 512>
        <<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16*)kv, headers_ptr, slices, in_blk, rows);
}

// Regression probe: evaluates the kernel-side `ds_exp` on a device array so
// the CPU mirror's replica can be asserted bit-identical over a full sweep.
__global__ void latent_exp_probe_kernel(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = latent_attn::ds_exp(in[i]);
}

extern "C" void run_latent_exp_probe(
    const float* in, float* out, int32_t n, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    latent_exp_probe_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

// Regression probe: the kernel-side RoPE trig (`rope_angle` + `ds_sincos`) for
// (pos, freq) pairs → interleaved (sin, cos) — asserted bit-identical to the
// CPU mirror.
__global__ void latent_sincos_probe_kernel(
    const int* pos, const float* freq, float* out, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r, s, c;
        int k;
        latent_attn::rope_angle(pos[i], freq[i], r, k);
        latent_attn::ds_sincos(r, k, s, c);
        out[2 * i] = s;
        out[2 * i + 1] = c;
    }
}

extern "C" void run_latent_sincos_probe(
    const int* pos, const float* freq, float* out, int32_t n, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    latent_sincos_probe_kernel<<<blocks, threads, 0, stream>>>(pos, freq, out, n);
}

// Build the factored RoPE cos/sin table for one frequency set (once per set at
// model load). `tab` holds (ROPE_HI_DIM + ROPE_LO_DIM) * n_freqs float2
// entries in the latent_common.cuh layout.
extern "C" void run_latent_rope_table_build(
    const float* freqs, float* tab, int32_t n_freqs, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int total =
        (latent_attn::ROPE_HI_DIM + latent_attn::ROPE_LO_DIM) * n_freqs;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    latent_attn::latent_rope_table_kernel<<<blocks, threads, 0, stream>>>(
        freqs, tab, n_freqs);
}

// Two-region position-free cache builder (B+D): nope [0,448) → int8 `nope_i8` +
// per-band amax `nope_scale` [G,14]; rope [448,512) → BF16 `rope_bf` [G,64].
extern "C" void run_latent_build_corpus_cache(
    const float* comp,   // [G, 512] f32 pre-RoPE (canonical)
    uint8_t* nope_i8,    // [G, 448] int8 out (nope)
    float* nope_scale,   // [G, 14] f32 out (per-nope-band amax)
    void* rope_bf,       // [G, 64] bf16 out (rope pre-rotation)
    int32_t g_lo,
    int32_t g_hi,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (g_hi <= g_lo) return;
    const int n = g_hi - g_lo;
    const int blocks = n < 2048 ? n : 2048;  // grid-stride caps the launch
    latent_attn::latent_build_corpus_cache_kernel<512, 64>
        <<<blocks, latent_attn::NPAL * 32, 0, stream>>>(
            comp, (int8_t*)nope_i8, nope_scale, (__nv_bfloat16*)rope_bf, g_lo, g_hi);
}

extern "C" void run_paged_latent_decode_bf16(
    const void* q_ptr,          // [slots, H, 512] bf16, pre-RoPE
    const uint8_t* headers_ptr, // SlotHeader[slots]
    void* o_ptr,                // [slots, H, 512] bf16, de-rotated final output
    const void* kv_new,         // [slots, 512] bf16, pre-RoPE latent
    const uint8_t* nope_i8,     // [G_total, 448] two-region cache: nope int8
    const float* nope_scale,    // [G_total, 14] per-nope-band scales
    const void* comp_rope,      // [G_total, 64] rope pre-rotation bf16
    const uint32_t* comp_idx,   // [slots, max_sel] ascending GIDs (0xFFFFFFFF pad)
    const uint32_t* comp_cnt,   // [slots] selected count
    const uint32_t* comp_pos,   // [G_total] assembled position per entry (rope-at-load)
    const uint32_t* q_pos,      // [slots] query position (explicit)
    const float* sinks,    // [H] per-head sink logits
    const float* rope_tab, // factored cos/sin table (latent_common.cuh layout)
    float* partial_acc,    // caller-owned [slots*H, num_splits, 512]
    float* partial_ml,     // caller-owned [slots*H, num_splits, 2]
    int32_t num_slots,
    int32_t n_q_head,
    float softmax_scale,
    int32_t window_size,
    int32_t max_sel,
    int32_t num_splits,
    int32_t commit_write_len, // 0 = skip the on-device write-len advance (wave)
    float* dbg,  // nullable stage-dump (mirror-oracle diagnostics)
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    latent_attn::launch_latent_decode<__nv_bfloat16, 512, 64>(
        (const __nv_bfloat16*)q_ptr, headers_ptr, (__nv_bfloat16*)o_ptr,
        (const __nv_bfloat16*)kv_new, (const int8_t*)nope_i8, nope_scale,
        (const __nv_bfloat16*)comp_rope, comp_idx, comp_cnt, comp_pos, q_pos,
        sinks, rope_tab, partial_acc, partial_ml,
        num_slots, n_q_head, softmax_scale, window_size, max_sel, num_splits,
        commit_write_len != 0, stream, dbg);
}

// SM count for the caller's split-factor policy.
extern "C" int32_t run_latent_sm_count() {
    return (int32_t)latent_attn::latent_sm_count();
}
