// The templated main batched sampling kernel.
//
// Holds the batched sampling kernel, templated on input type and mode, which
// runs penalties, the tiled sampling pass and the chosen sampler for every
// sequence in one launch.
// It is one 25 KB function, so it stays whole.

// ============================================================================
// Main Batched Sampling Kernel (Templated for Input Type and Mode)
// ============================================================================

// Template parameters for compile-time optimization:
// - T: Input logit type (float, half, __nv_bfloat16, __nv_fp8_*)
// - MAX_K: Maximum top-k value supported
// - THREADS: Threads per block
// - USE_PENALTIES: When false, all standard penalty code is eliminated (saves tile_buffer 4KB)
// - USE_DRY: When false, DRY penalty cache is eliminated (saves 4KB)
// - USE_TOP_P: When false, nucleus filtering loop is eliminated
//
// Shared memory savings:
// - USE_PENALTIES=false: eliminates tile_buffer (4KB)
// - USE_DRY=false: eliminates dry_cache (4KB)
// - Both false: save 8KB → better occupancy for no-penalty/argmax paths

// Main batched sampling kernel - takes all parameters directly (no cudaMalloc needed)
// All GPU pointers are already on device (from Tensors), scalars are passed by value
template <typename T, int MAX_K, int THREADS, bool USE_PENALTIES, bool USE_DRY, bool USE_TOP_P>
__global__ void __launch_bounds__(THREADS, USE_PENALTIES ? 1 : 2)
batched_penalty_sampling_kernel(
    // Input logits from LM head matmul
    const T* __restrict__ logits,           // [batch_size, vocab_size]
    int vocab_size,
    // Tokens a row may produce. A checkpoint pads its output projection past
    // the tokenizer's last id; those logits come from rows no token was ever
    // trained through and must carry no probability — not in the argmax, the
    // sample, or the typical-acceptance entropy. `vocab_size` stays the row
    // stride; only the walk over candidates stops here.
    int live_vocab,
    
    // Penalty scalars (passed by value, CUDA copies to constant memory)
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
    // Cross-turn penalty
    float cross_turn_penalty,
    const int32_t* __restrict__ cross_turn_counts,  // [batch_size, vocab_size] or null
    // Per-sequence current lengths (for dynamic EOS ramp)
    const int32_t* __restrict__ current_lens,       // [batch_size] or null
    // Segment-close boost
    float segment_close_boost,
    int32_t segment_close_token_id,
    int32_t segment_close_ramp_start,
    int32_t segment_close_ramp_len,
    float segment_close_max_multiplier,
    const int32_t* __restrict__ segment_lens,      // [batch_size] or null
    // Per-sequence DRY window: the count of trailing `recent_tokens` that
    // belong to the free-text value being written. The caller sets it; it is
    // 0 outside a value. Null / 0 => DRY off for that sequence.
    const int32_t* __restrict__ dry_lens,          // [batch_size] or null
    // Per-token DRY_REFERENCE/DRY_JOINS flags: a continuation inside a
    // reference is never penalised by DRY. Null => no token is exempt.
    const uint8_t* __restrict__ dry_ref_flags,     // [vocab_size] or null
    // Added to `temperature` for sequences inside a segment (segment_lens[seq] > 0).
    float segment_temp_boost,
    // Token suppression: subtract `suppress_penalties[seq]` from each
    // `suppress_tokens` logit while sequence `seq` is inside a segment.
    const int32_t* __restrict__ suppress_tokens,    // [suppress_count] (shared) or null
    int32_t suppress_count,
    const float* __restrict__ suppress_penalties,   // [batch_size] or null

    // Penalty GPU pointers (already on device from Tensors)
    const int32_t* __restrict__ token_counts,     // [batch_size, vocab_size] or null
    const int32_t* __restrict__ banned_tokens,    // [batch_size, banned_per_seq] or [num_banned] or null
    int32_t num_banned_tokens,
    int32_t banned_tokens_per_seq,
    
    // Recent tokens for repeat penalty (per-sequence, already on device)
    const int32_t* __restrict__ recent_tokens,    // [batch_size * max_recent_len] or null
    const int32_t* __restrict__ recent_lens,      // [batch_size] or null
    int32_t max_recent_len,
    
    // Sampling configuration
    float temperature,
    int top_k,
    float top_p,

    // Output
    uint32_t* __restrict__ output_tokens,    // [batch_size]
    // [batch_size] or null: 1 where the row's live logits hold a NaN or a
    // +inf (`logit_not_finite`), else 0 — every row's word written.
    uint32_t* __restrict__ nonfinite_rows,

    // RNG
    uint64_t seed,
    uint64_t* __restrict__ rng_offsets,      // [batch_size], updated in-place

    // Per-sequence dials — [batch_size] or null. When non-null, EVERY dial above
    // (temperature, top-k/p, penalties, EOS/segment ramps) is overridden per row
    // from `seq_dials[blockIdx.x]`; the scalar arguments become defaults for the
    // null case only. See `SeqDials`.
    const SeqDials* __restrict__ seq_dials
) {
    // Shared memory with conditional allocation based on feature flags
    // USE_PENALTIES controls tile_buffer (4KB), USE_DRY controls dry_cache (4KB)
    __shared__ SamplingSharedMem<MAX_K, THREADS, USE_PENALTIES, USE_DRY, true> smem;
    
    // Dynamic shared memory for bitset (allocated based on vocab_size at launch)
    extern __shared__ uint32_t recent_bitset[];
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Safety: validate required pointers
    // Note: rng_offsets can be null for argmax (temperature <= 0) since RNG isn't used
    if (output_tokens == nullptr || logits == nullptr) {
        if (tid == 0 && output_tokens != nullptr) {
            output_tokens[batch_idx] = 0;  // Fallback to token 0
        }
        if (tid == 0 && nonfinite_rows != nullptr) {
            nonfinite_rows[batch_idx] = 0;
        }
        return;
    }

    // **Per-sequence dials.** The scalar dial arguments are mutable value params,
    // so overriding them here — before any of them is read to build the penalty
    // params or drive sampling — makes every downstream use per-row with no other
    // change to this kernel. A null `seq_dials` leaves the scalars untouched
    // (legacy shared-config behaviour). This is the whole of the per-sequence
    // sampling fix; `eos_token_id` and vocab size stay scalar because they do
    // not vary across the wave, but `segment_close_token_id` does (a narrator
    // row carries -1 beside a deliberating row's `</think>`), so it is per-row.
    // A bound past the row could only read beyond it.
    live_vocab = min(live_vocab, vocab_size);
    // A verify row carries the draft it tests (see `SeqDials`).
    int32_t typical_draft = -1;
    float typical_eps = 0.0f;
    float typical_delta = 0.0f;
    // The logits row this block samples (see `SeqDials::logits_row`); the
    // block's own index when no dials are given.
    int32_t logits_row = batch_idx;
    if (seq_dials != nullptr) {
        const SeqDials d = seq_dials[batch_idx];
        logits_row = d.logits_row;
        typical_draft = d.typical_draft;
        typical_eps = d.typical_eps;
        typical_delta = d.typical_delta;
        temperature = d.temperature;
        top_k = d.top_k;
        top_p = d.top_p;
        repeat_penalty = d.repeat_penalty;
        frequency_penalty = d.frequency_penalty;
        presence_penalty = d.presence_penalty;
        dry_multiplier = d.dry_multiplier;
        dry_base = d.dry_base;
        dry_allowed_length = d.dry_allowed_length;
        dry_range = d.dry_range;
        eos_boost = d.eos_boost;
        eos_ramp_start = d.eos_ramp_start;
        eos_ramp_len = d.eos_ramp_len;
        eos_boost_max_multiplier = d.eos_boost_max_multiplier;
        cross_turn_penalty = d.cross_turn_penalty;
        segment_close_boost = d.segment_close_boost;
        segment_close_token_id = d.segment_close_token_id;
        segment_close_ramp_start = d.segment_close_ramp_start;
        segment_close_ramp_len = d.segment_close_ramp_len;
        segment_close_max_multiplier = d.segment_close_max_multiplier;
        segment_temp_boost = d.segment_temp_boost;
    }
    
    const T* my_logits = logits + (size_t)logits_row * vocab_size;

    // Calculate bitset words needed for this vocab size
    const int bitset_words = (vocab_size + 31) / 32;
    
    // Build PenaltyParams struct on registers from scalar parameters
    // This is free - compiler optimizes it to direct register usage
    PenaltyParams penalty_params_local;
    penalty_params_local.repeat_penalty = repeat_penalty;
    penalty_params_local.frequency_penalty = frequency_penalty;
    penalty_params_local.presence_penalty = presence_penalty;
    penalty_params_local.dry_multiplier = dry_multiplier;
    penalty_params_local.dry_base = dry_base;
    penalty_params_local.dry_allowed_length = dry_allowed_length;
    penalty_params_local.dry_range = dry_range;
    penalty_params_local.eos_boost = eos_boost;
    penalty_params_local.eos_token_id = eos_token_id;
    penalty_params_local.eos_ramp_start = eos_ramp_start;
    penalty_params_local.eos_ramp_len = eos_ramp_len;
    penalty_params_local.eos_boost_max_multiplier = eos_boost_max_multiplier;
    penalty_params_local.current_len = (current_lens != nullptr) ? current_lens[batch_idx] : 0;
    penalty_params_local.segment_close_boost = segment_close_boost;
    penalty_params_local.segment_close_token_id = segment_close_token_id;
    penalty_params_local.segment_close_ramp_start = segment_close_ramp_start;
    penalty_params_local.segment_close_ramp_len = segment_close_ramp_len;
    penalty_params_local.segment_close_max_multiplier = segment_close_max_multiplier;
    penalty_params_local.segment_len = (segment_lens != nullptr) ? segment_lens[batch_idx] : 0;
    penalty_params_local.cross_turn_penalty = cross_turn_penalty;
    penalty_params_local.cross_turn_counts = cross_turn_counts;
    penalty_params_local.token_counts = token_counts;
    penalty_params_local.vocab_size = vocab_size;
    penalty_params_local.banned_tokens = banned_tokens;
    penalty_params_local.num_banned_tokens = num_banned_tokens;
    penalty_params_local.banned_tokens_per_seq = banned_tokens_per_seq;
    penalty_params_local.allowed_bits = nullptr;
    if (seq_dials != nullptr) {
        const SeqDials& d = seq_dials[batch_idx];
        penalty_params_local.allowed_bits = reinterpret_cast<const uint32_t*>(
            (static_cast<uint64_t>(d.allowed_bits_hi) << 32) | d.allowed_bits_lo);
    }
    penalty_params_local.suppress_tokens = suppress_tokens;
    penalty_params_local.suppress_count = suppress_count;
    // Per-sequence suppression strength; gated to segments just below.
    penalty_params_local.suppress_penalty = 0.0f;

    // =========================================================
    // In-segment steering (per-sequence, since each block = one sequence).
    // While this sequence is inside a segment (segment_lens[seq] > 0):
    //   - sample a touch hotter (temperature += segment_temp_boost)
    //   - activate token suppression (the in-segment ceiling lever).
    // Outside a segment both revert to the batch-wide values.
    // =========================================================
    const bool in_segment = (segment_lens != nullptr) && (segment_lens[batch_idx] > 0);
    if (in_segment) {
        temperature += segment_temp_boost;
        // Activate token suppression for this sequence: subtract the per-sequence
        // penalty from every suppress-token logit (applied in
        // apply_penalties_branchless). Outside a segment this stays 0.
        if (suppress_penalties != nullptr) {
            penalty_params_local.suppress_penalty = suppress_penalties[batch_idx];
        }
    }

    // =========================================================
    // DRY gating (per-sequence), independent of the think segment.
    // DRY is scoped via `dry_lens[seq]` — the count of trailing recent tokens
    // that belong to the window the caller chose (the free-text value being
    // written), so it breaks a loop inside that window without penalizing
    // verbatim reproduction of anything outside it. `dry_lens[seq] == 0`
    // disables DRY for that sequence without touching the segment-driven temp
    // boost / suppression above.
    // =========================================================
    const int32_t dry_span_len = (dry_lens != nullptr) ? dry_lens[batch_idx] : 0;
    const bool dry_on = (dry_span_len > 0);
    if (!dry_on) {
        // Disable DRY for this sequence: zero the multiplier the precompute and
        // cached-lookup paths key off.  Shared memory for the cache is still
        // allocated (USE_DRY is a batch-wide template param), but no penalties
        // are produced for this sequence.
        dry_multiplier = 0.0f;
        penalty_params_local.dry_multiplier = 0.0f;
    }

    // =========================================================
    // STEP 1: Build recent token bitset (if penalties enabled)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (recent_tokens != nullptr && recent_lens != nullptr) {
            // Get this sequence's recent tokens (oldest-first; newest at the end).
            const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
            int my_recent_len = recent_lens[batch_idx];

            // Repeat-penalty window: segment-scoped. While inside a segment the
            // repeat penalty must only see the tokens generated inside the
            // current segment — never the prompt or prior turns that precede the
            // segment-open token. The segment spans the last `segment_len`
            // tokens, so restrict the window to that suffix (`recent_tokens` is
            // newest-at-the-end). Outside a segment, keep the full window.
            const int32_t* rep_recent = my_recent;
            int rep_recent_len = my_recent_len;
            if (in_segment) {
                int effective_len = min(segment_lens[batch_idx], my_recent_len);
                rep_recent = my_recent + (my_recent_len - effective_len);
                rep_recent_len = effective_len;
            }

            build_recent_bitset<THREADS>(
                recent_bitset,
                rep_recent,
                rep_recent_len,
                bitset_words
            );

            // Precompute DRY penalties (O(n²) once, then O(1) lookup per token).
            // DRY has its OWN window: the last `dry_span_len` recent tokens (the
            // current structural span), independent of the segment scoping above.
            // Only when USE_DRY template param is enabled.
            if constexpr (USE_DRY) {
                int dry_len = dry_on ? min(dry_span_len, my_recent_len) : 0;
                if (dry_multiplier != 0.0f && dry_len >= 2) {
                    const int32_t* dry_recent = my_recent + (my_recent_len - dry_len);
                    int eff_dry_range = (dry_range > 0) ? min(dry_range, dry_len) : dry_len;
                    precompute_dry_penalties<THREADS>(
                        dry_recent, dry_len,
                        dry_multiplier, dry_base, dry_allowed_length, eff_dry_range,
                        dry_ref_flags,
                        smem.dry_cache.as_ptr()
                    );
                } else {
                    // No DRY - clear cache
                    if (tid == 0) {
                        smem.dry_cache.num_entries = 0;
                    }
                    __syncthreads();
                }
            }
        } else {
            // Clear bitset
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            // Clear DRY cache (only if enabled)
            if constexpr (USE_DRY) {
                if (tid == 0) {
                    smem.dry_cache.num_entries = 0;
                }
            }
            __syncthreads();
        }
    } else {
        // No penalties - just clear bitset (minimal cost, ensures valid reads)
        for (int i = tid; i < bitset_words; i += THREADS) {
            recent_bitset[i] = 0;
        }
        __syncthreads();
    }
    
    // Pointer to DRY cache (use cached version for O(1) lookup)
    // When USE_DRY=false, dry_cache.as_ptr() returns nullptr
    const DryPenaltyCache* dry_cache_ptr = USE_DRY ? smem.dry_cache.as_ptr() : nullptr;
    
    // Pointer to use - null when penalties disabled for cleaner downstream code
    const PenaltyParams* effective_penalties = USE_PENALTIES ? &penalty_params_local : nullptr;

    // =========================================================
    // STEP 2: Early exit for argmax (temperature <= 0)
    // =========================================================
    if (!(temperature > 0.f)) {  // Handles NaN, <=0, -inf
        bool not_finite = false;
        int best = branchless_argmax_typed<T, THREADS>(
            my_logits, live_vocab, batch_idx,
            effective_penalties, recent_bitset, dry_cache_ptr,
            smem.reduction_max, smem.reduction_idx, not_finite
        );

        if (tid == 0) {
            output_tokens[batch_idx] = best;
            if (nonfinite_rows != nullptr) {
                nonfinite_rows[batch_idx] = not_finite ? 1u : 0u;
            }
        }
        return;
    }

    const float inv_temp = 1.0f / temperature;
    const int k = top_k > 0 ? min(top_k, MAX_K) : MAX_K;

    float global_max, global_sum;
    int num_topk;
    bool not_finite = false;
    
    // =========================================================
    // Processing strategy based on penalties and top-k
    // =========================================================
    if constexpr (USE_PENALTIES) {
        // =========================================================
        // TOP-K + PENALTY PATH: 5 passes instead of 8. Radix select on
        // RAW LOGITS (no exp), then local softmax over only the ~k
        // collected candidates in shared memory.
        //
        // This runs for `top_k <= 0` as well, where `k` is MAX_K. The
        // threshold-based collector that used to serve that case gathered on
        // `prob >= 0.0f` — true for every token — and kept whichever
        // `atomicAdd` tickets landed under `k`. The vocabulary is walked in
        // ascending token order, so the first tile claimed every slot and the
        // candidate set was tokens `0..k`: the byte-level and punctuation range
        // of a BPE vocabulary. "No top-k limit" therefore meant "only the
        // lowest token ids", and a decode whose real answer lived above that
        // window could not emit it — live, the whole response collapsed onto
        // token 0, `!` in the Qwen vocabulary. Radix select is unbiased, so
        // `top_k <= 0` now honestly means "the top MAX_K by logit".
        // =========================================================

        // Step 1: Radix select on raw (penalized) logits — 4 vocab passes, NO exp()
        float logit_threshold = radix_select_logit_threshold<T, THREADS>(
            my_logits, live_vocab, batch_idx, k,
            effective_penalties, recent_bitset, dry_cache_ptr,
            smem.radix_histogram, smem.reduction_max, not_finite
        );

        // Broadcast threshold to ensure all threads agree
        if (tid == 0) smem.global_max = logit_threshold;
        __syncthreads();
        logit_threshold = smem.global_max;

        // Step 2: Collect + local softmax — 1 vocab pass, probs computed in smem
        num_topk = collect_and_locally_normalize<T, THREADS>(
            my_logits, live_vocab, batch_idx, logit_threshold, inv_temp,
            effective_penalties, recent_bitset, dry_cache_ptr,
            smem.topk_vals, smem.topk_idxs, k, &smem.topk_count,
            smem.reduction_max
        );

        // Probabilities are already locally normalized (sum to 1.0)
        if (tid == 0) {
            smem.global_sum = 1.0f;
            smem.best_idx = (num_topk > 0) ? smem.topk_idxs[0] : 0;
        }
        __syncthreads();
    } else {
        // =========================================================
        // NO-PENALTY PATH
        // =========================================================
        
        // =========================================================
        // TOP-K PATH: 5 passes instead of 8, no exp(). Radix select on
        // raw logits, then local softmax over ~k items.
        //
        // As on the penalty side above, this runs for `top_k <= 0` too,
        // where `k` is MAX_K. The global-softmax-then-threshold-collect
        // branch that used to serve that case gathered on `prob >= 0.0f`,
        // which every token satisfies, and kept whichever `atomicAdd`
        // tickets landed under `k` — the vocabulary is walked in ascending
        // token order, so the candidate set was tokens `0..k`. Sampling
        // could then only ever return a low token id.
        // =========================================================

        // Step 1: Radix select on raw logits — 4 vocab passes, NO exp()
        float logit_threshold = radix_select_logit_threshold<T, THREADS>(
            my_logits, live_vocab, batch_idx, k,
            nullptr, recent_bitset, nullptr,  // No penalties, no DRY
            smem.radix_histogram, smem.reduction_max, not_finite
        );

        // Broadcast threshold
        if (tid == 0) smem.global_max = logit_threshold;
        __syncthreads();
        logit_threshold = smem.global_max;

        // Step 2: Collect + local softmax — 1 vocab pass
        num_topk = collect_and_locally_normalize<T, THREADS>(
            my_logits, live_vocab, batch_idx, logit_threshold, inv_temp,
            nullptr, recent_bitset, nullptr,
            smem.topk_vals, smem.topk_idxs, k, &smem.topk_count,
            smem.reduction_max
        );

        // Probabilities are already locally normalized (sum to 1.0)
        if (tid == 0) {
            smem.global_sum = 1.0f;
            smem.best_idx = (num_topk > 0) ? smem.topk_idxs[0] : 0;
        }
        __syncthreads();
    }
    
    const float inv_sum = 1.0f / smem.global_sum;
    
    if (num_topk <= 0) {
        if (tid == 0) {
            smem.topk_vals[0] = 1.0f;
            smem.topk_idxs[0] = smem.best_idx;
        }
        num_topk = 1;
    }
    __syncthreads();
    
    bitonic_sort_descending(smem.topk_vals, smem.topk_idxs, num_topk);
    __syncthreads();
    
    int nucleus_size = num_topk;
    
    if constexpr (USE_TOP_P) {
        // NOTE on top-p normalization semantics:
        // When top-k is active, collect_and_locally_normalize has already
        // normalized probabilities over ONLY the top-k set (local normalization).
        // inv_sum was set to 1.0f so topk_vals are already proper probabilities
        // that sum to 1.0 within the top-k set.  The cumsum here accumulates
        // those local probabilities directly — this matches vLLM / HuggingFace
        // behavior where top-p is applied AFTER top-k filtering.
        //
        // When top-k is NOT active (no-top-k tiled path), probabilities come
        // from global softmax with inv_sum = 1/global_sum, so the cumsum
        // operates over globally normalized probabilities.
        if (top_p < 1.0f && tid == 0) {
            float cumsum = 0.f;
            for (int i = 0; i < num_topk; i++) {
                cumsum += smem.topk_vals[i] * inv_sum;
                if (cumsum >= top_p) {
                    nucleus_size = i + 1;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        // rng_offsets should never be null when temperature > 0, but check defensively
        uint64_t offset = rng_offsets ? rng_offsets[batch_idx] : 0;
        if (rng_offsets) {
            rng_offsets[batch_idx] = offset + 1;
        }
        
        const int candidates = nucleus_size > 0 ? nucleus_size : 1;
        uint32_t token = sample_from_topk(
            smem.topk_vals, smem.topk_idxs, candidates, seed, offset
        );
        // A verify row commits its draft when the distribution gives it
        // enough mass, and its sample otherwise. The sample is drawn either
        // way, so the row's RNG stream advances exactly as a plain row's.
        if (typical_draft >= 0 && typical_accepts(
                smem.topk_vals, smem.topk_idxs, candidates,
                typical_draft, typical_eps, typical_delta)) {
            token = (uint32_t)typical_draft;
        }
        output_tokens[batch_idx] = token;
        if (nonfinite_rows != nullptr) {
            nonfinite_rows[batch_idx] = not_finite ? 1u : 0u;
        }
    }
}
