// Batched sampling kernel - FP32 instantiation
#include "batched_sampling.cuh"

using namespace batched_sampling;

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
    // EOT (end-of-thinking) boost
    float eot_boost,
    int32_t eot_token_id,
    int32_t eot_ramp_start,
    int32_t eot_ramp_len,
    float eot_boost_max_multiplier,
    const int32_t* thinking_lens,
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
    launch_batched_sampling_typed<float>(
        logits, batch_size, vocab_size,
        repeat_penalty, frequency_penalty, presence_penalty,
        dry_multiplier, dry_base, dry_allowed_length, dry_range,
        eos_boost, eos_token_id,
        eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
        cross_turn_penalty, cross_turn_counts, current_lens,
        eot_boost, eot_token_id, eot_ramp_start, eot_ramp_len, eot_boost_max_multiplier, thinking_lens,
        token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
        recent_tokens, recent_lens, max_recent_len,
        stencil, stencil_size,
        temperature, top_k, top_p,
        output_tokens, seed, rng_offsets
    );
}
