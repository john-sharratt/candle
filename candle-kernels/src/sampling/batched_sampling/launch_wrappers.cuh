// Shared-memory sizing and the dispatch that selects a kernel variant from
// runtime flags.
//
// Holds `raise_dynamic_smem_for`, which sets the dynamic shared-memory limit a
// variant needs, and `dispatch_batched_sampling`, which selects the variant
// from runtime flags.
// Holds `launch_batched_sampling_typed`, which auto-detects modes and stencil
// support before launching.

// ============================================================================
// Kernel Launch Wrappers with Compile-Time Dispatch
// ============================================================================

/// Raise a sampling kernel's dynamic shared-memory ceiling to exactly what this
/// vocabulary needs — and ONLY when the default ceiling is too small.
///
/// The recent-token bitset is dynamic and VOCAB-SIZED — `ceil(vocab/32)*4`
/// bytes — and the kernel's static block is ~19 KiB once the DRY cache is in
/// it. A block gets 48 KiB by default, so the pair crosses the ceiling at a
/// vocabulary of roughly 240,000: measured, a 28.9 KiB bitset launches and a
/// 29.3 KiB one does not. Qwen3.6-35B-A3B's 248,320-token vocabulary is past
/// that line, and the rejection was invisible — this translation unit's entry
/// points return `void`, so the caller kept its zero-initialised output buffer
/// and every row "sampled" token 0, which is `!` in the Qwen vocabularies.
///
/// **Ask for the need, not the device maximum.** Opting into the full allowance
/// pushes the kernel past the 48 KiB line unconditionally, and crossing it makes
/// the driver shift the SM's L1/shared carveout toward shared memory. That guts
/// L1 for a kernel whose whole job is scanning a 248,320-entry vocabulary, and
/// it is charged on every sampling call — including the small-vocabulary ones
/// that never needed a byte of it. Measured on the 3.6-35B forward gate: bulk
/// throughput fell from ~6,900 t/s to ~2,150, a 3.3x loss, restored in full by
/// sizing the request. A launch that already fits keeps the default carveout and
/// its L1.
///
/// The stored high-water mark only ever GROWS, so a small-vocabulary caller can
/// never lower a ceiling a large-vocabulary launch on another thread is relying
/// on — the raise is monotonic, never an assignment.
///
/// The mark is per-`T`, not per-device: a process driving two GPUs would treat
/// a raise applied on one as applied on both. Every deployment here is one
/// device per process, and widening it to a `device -> bytes` map is the fix if
/// that stops being true.
template <typename T>
static inline void raise_dynamic_smem_for(size_t dynamic_bytes) {
    static size_t applied = 0;
    if (dynamic_bytes <= applied) {
        return;
    }

    const void* kernels[] = {
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, true>,
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, false>,
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, true>,
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, false>,
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, true>,
        (const void*)batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, false>,
    };

    // Per-block default, below which the carveout is left alone entirely.
    constexpr size_t DEFAULT_BLOCK_SMEM = 48 * 1024;
    // **Every variant that needed raising, or the mark does not move.**
    //
    // Recording the high-water mark when merely SOME variant was raised is
    // worse than not recording it: the early-out at the top then skips the
    // whole function, so a variant whose `cudaFuncSetAttribute` failed is never
    // retried. Its launch is rejected for exceeding the block ceiling, the
    // `void` entry point reports nothing, and the caller reads back the
    // `output_tokens` it passed in — which is token 0 for every sequence. That
    // is the silent degenerate-decode signature this function exists to
    // prevent, reintroduced by its own bookkeeping.
    bool all_ok = true;
    for (const void* k : kernels) {
        cudaFuncAttributes attrs{};
        if (cudaFuncGetAttributes(&attrs, k) != cudaSuccess) {
            all_ok = false;
            continue;
        }
        if (attrs.sharedSizeBytes + dynamic_bytes <= DEFAULT_BLOCK_SMEM) {
            continue; // Fits as-is; touching the attribute would only cost L1.
        }
        // A device that declines leaves the previous limit in place; the launch
        // then fails as it did before rather than reading garbage.
        if (cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(dynamic_bytes))
            != cudaSuccess) {
            all_ok = false;
        }
    }
    if (all_ok) {
        applied = dynamic_bytes;
    }
}

// Internal dispatch helper - selects kernel variant based on runtime flags
template <typename T>
inline void dispatch_batched_sampling(
    const T* logits,
    int batch_size,
    int vocab_size,
    int live_vocab,
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
    const uint8_t* dry_ref_flags,
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
    // Sampling params
    float temperature,
    int top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    uint32_t* nonfinite_rows,
    uint64_t seed,
    uint64_t* rng_offsets,
    // Per-sequence dials ([batch_size] or null) — see SeqDials.
    const SeqDials* seq_dials,
    // Dispatch flags
    bool use_penalties,
    bool use_top_p,
    // The stream the launch goes on — the caller's, never the null stream
    cudaStream_t stream
) {

    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);

    // Dynamic shared memory for bitset: ceil(vocab_size / 32) words * 4 bytes
    const size_t bitset_bytes = ((vocab_size + 31) / 32) * sizeof(uint32_t);

    // 4-way dispatch based on penalties and top_p
    // USE_DRY is set to true when penalties are enabled AND dry_multiplier != 0
    // This allows eliminating 4KB shared memory when DRY is not used.
    // With per-sequence dials the scalar `dry_multiplier` says nothing about what
    // individual rows want, so the DRY template is forced on — the kernel still
    // gates DRY per row on that row's own `dry_multiplier`/`dry_lens`, so a row
    // that wants none simply skips the precompute.
    const bool use_dry =
        (seq_dials != nullptr) || (use_penalties && (dry_multiplier != 0.0f));

    // A large vocabulary's recent-token bitset does not fit under the default
    // 48 KiB block ceiling alongside this kernel's static block; opt into the
    // device maximum before any launch can be rejected for it.
    // Raise the ceiling only if this vocabulary's bitset does not already fit.
    raise_dynamic_smem_for<T>(bitset_bytes);

    if (use_penalties && use_dry && use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    } else if (use_penalties && use_dry && !use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, true, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    } else if (use_penalties && !use_dry && use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    } else if (use_penalties && !use_dry && !use_top_p) {
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, true, false, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    } else if (!use_penalties && use_top_p) {
        // No penalties implies no DRY either
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    } else {
        // No penalties, no DRY, no top_p - fastest path
        batched_penalty_sampling_kernel<T, MAX_TOP_K, THREADS_PER_BLOCK, false, false, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size, live_vocab,
                repeat_penalty, frequency_penalty, presence_penalty,
                dry_multiplier, dry_base, dry_allowed_length, dry_range,
                eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
                cross_turn_penalty, cross_turn_counts, current_lens,
                segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
                dry_lens, dry_ref_flags,
                segment_temp_boost,
                suppress_tokens, suppress_count, suppress_penalties,
                token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets,
                seq_dials
            );
    }
}

// Templated launch wrapper with auto-detection of modes and stencil support
template <typename T>
inline void launch_batched_sampling_typed(
    const T* logits,
    int batch_size,
    int vocab_size,
    // Tokens a row may produce; see `batched_penalty_sampling_kernel`. The
    // stencil kernel reads only its allow-list, which names real tokens.
    int live_vocab,
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
    const uint8_t* dry_ref_flags,
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
    // Stencil (constrained vocabulary) - pass nullptr or stencil_size=0 to disable
    const int32_t* stencil,
    int32_t stencil_size,
    // Sampling params
    float temperature,
    int top_k,
    float top_p,
    // Output
    uint32_t* output_tokens,
    // [batch_size] or null: each row's non-finite verdict (`logit_not_finite`).
    uint32_t* nonfinite_rows,
    uint64_t seed,
    uint64_t* rng_offsets,
    // Per-sequence dials ([batch_size] or null) — see SeqDials.
    const SeqDials* seq_dials,
    // The stream the launch goes on — the caller's, never the null stream
    cudaStream_t stream
) {
    // Auto-detect whether penalties are needed
    bool use_penalties = (repeat_penalty != 1.0f) ||
                         (frequency_penalty != 0.0f) ||
                         (presence_penalty != 0.0f) ||
                         (cross_turn_penalty != 0.0f) ||
                         (dry_multiplier != 0.0f) ||
                         (eos_boost != 0.0f) ||
                         (segment_close_boost != 0.0f) ||
                         (num_banned_tokens > 0) ||
                         (banned_tokens_per_seq > 0) ||
                         // Token suppression needs the penalty path.
                         (suppress_tokens != nullptr && suppress_count > 0 &&
                          suppress_penalties != nullptr);
    bool use_top_p = (top_p < 1.0f);
    // With per-sequence dials the scalar detection above says nothing about what
    // individual rows want, so force the full-featured kernel variant on: its
    // penalty and top-p paths are no-ops for a row whose own dials are neutral,
    // and every row then samples on exactly its own config.
    if (seq_dials != nullptr) {
        use_penalties = true;
        use_top_p = true;
    }

    // Use optimized stencil kernel when stencil is provided and reasonably small
    // For stencils up to MAX_STENCIL_SIZE (1024), direct iteration is faster
    // Note: Stencil kernel doesn't support DRY penalty yet - fall through if DRY enabled
    if (stencil != nullptr && stencil_size > 0 && stencil_size <= MAX_STENCIL_SIZE && dry_multiplier == 0.0f) {
        dispatch_stencil_sampling<T>(
            logits, batch_size, vocab_size,
            stencil, stencil_size,
            repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
            token_counts, banned_tokens, num_banned_tokens,
            recent_tokens, recent_lens, max_recent_len,
            temperature, top_k, top_p,
            output_tokens, nonfinite_rows, seed, rng_offsets,
            use_penalties,
            stream
        );
    } else {
        // Fall back to full vocabulary kernel
        dispatch_batched_sampling<T>(
            logits, batch_size, vocab_size, live_vocab,
            repeat_penalty, frequency_penalty, presence_penalty,
            dry_multiplier, dry_base, dry_allowed_length, dry_range,
            eos_boost, eos_token_id, eos_ramp_start, eos_ramp_len, eos_boost_max_multiplier,
            cross_turn_penalty, cross_turn_counts, current_lens,
            segment_close_boost, segment_close_token_id, segment_close_ramp_start, segment_close_ramp_len, segment_close_max_multiplier, segment_lens,
            dry_lens, dry_ref_flags,
            segment_temp_boost,
            suppress_tokens, suppress_count, suppress_penalties,
            token_counts, banned_tokens, num_banned_tokens, banned_tokens_per_seq,
            recent_tokens, recent_lens, max_recent_len,
            temperature, top_k, top_p,
            output_tokens, nonfinite_rows, seed, rng_offsets,
            seq_dials,
            use_penalties, use_top_p,
            stream
        );
    }
}
