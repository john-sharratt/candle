// C API wrapper for batched penalty and sampling kernel
// This provides the extern "C" interface for Rust FFI
// 
// Key design: NO cudaMalloc/cudaMemcpy needed!
// - All scalar parameters are passed directly to the kernel
// - All GPU pointers (logits, token_counts, etc.) come from Rust Tensors
//   which are already on the GPU
// - CUDA kernel launch uses default stream (0), like prefill kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Forward declarations of typed kernel entry points (defined in separate .cu files)
// Note: No cudaStream_t parameter - kernels use default stream internally
extern "C" void run_batched_sampling_f32(
    const float* logits,
    int32_t batch_size,
    int32_t vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    const int32_t* dry_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary)
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int32_t top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
);

extern "C" void run_batched_sampling_f16(
    const half* logits,
    int32_t batch_size,
    int32_t vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    const int32_t* dry_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary)
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int32_t top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
);

extern "C" void run_batched_sampling_fp8_e4m3(
    const __nv_fp8_e4m3* logits,
    int32_t batch_size,
    int32_t vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    const int32_t* dry_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary)
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int32_t top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
);

extern "C" void run_batched_sampling_bf16(
    const __nv_bfloat16* logits,
    int32_t batch_size,
    int32_t vocab_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    float eos_boost,
    int32_t eos_token_id,
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    const int32_t* dry_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary)
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int32_t top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
);

// ============================================================================
// Unified Dispatcher
// ============================================================================
// dtype: 0 = f32, 1 = f16, 2 = bf16

extern "C" void run_batched_sampling(
    const void* logits,
    int32_t batch_size,
    int32_t vocab_size,
    int32_t dtype,
    // Sampling params
    float temperature,
    int32_t top_k,
    float top_p,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    // DRY penalty params
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_range,
    // EOS boost
    float eos_boost,
    int32_t eos_token_id,
    // Dynamic EOS boost ramp + cross-turn penalty
    int32_t eos_ramp_start,
    int32_t eos_ramp_len,
    float eos_boost_max_multiplier,
    float cross_turn_penalty,
    const int32_t* cross_turn_counts,
    const int32_t* current_lens,
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* segment_lens,
    const int32_t* dry_lens,
    float segment_temp_boost,
    const int32_t* suppress_tokens,
    int32_t suppress_count,
    const float* suppress_penalties,
    // Penalty GPU pointers (already on device)
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    // Recent tokens (already on device)
    const int32_t* recent_tokens,
    const int32_t* recent_lens,
    int32_t max_recent_len,
    // Stencil (constrained vocabulary) - pass nullptr/0 to disable
    const int32_t* stencil,
    int32_t stencil_size,
    // Output
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
) {
    switch (dtype) {
        case 0: // f32
            run_batched_sampling_f32(
                reinterpret_cast<const float*>(logits), batch_size, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id,
                eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                stencil, stencil_size,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
            break;
        case 1: // f16
            run_batched_sampling_f16(
                reinterpret_cast<const half*>(logits), batch_size, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id,
                eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                stencil, stencil_size,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
            break;
        case 2: // bf16
            run_batched_sampling_bf16(
                reinterpret_cast<const __nv_bfloat16*>(logits), batch_size, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id,
                eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                stencil, stencil_size,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
            break;
        case 3: // fp8_e4m3
            run_batched_sampling_fp8_e4m3(
                reinterpret_cast<const __nv_fp8_e4m3*>(logits), batch_size, vocab_size,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id,
                eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                stencil, stencil_size,
                temperature, top_k, top_p,
                output_tokens, seed, rng_offsets
            );
            break;
    }
}

