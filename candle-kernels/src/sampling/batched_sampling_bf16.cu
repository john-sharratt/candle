// Batched sampling kernel - BF16 instantiation
#include "batched_sampling.cuh"

using namespace batched_sampling;

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
) {
    launch_batched_sampling_typed<__nv_bfloat16>(
        logits, batch_size, vocab_size,
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
}
