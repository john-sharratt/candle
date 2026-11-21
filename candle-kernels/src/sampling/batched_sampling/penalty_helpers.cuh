// Recent-token bitset, DRY penalties, and branchless penalty application.
//
// Builds the recent-token bitset that gives O(1) lookup, and the precomputed
// DRY penalty cache that `lookup_dry_penalty` reads branchlessly to avoid warp
// divergence.
// Holds `apply_penalties_branchless` and its cached and legacy DRY variants,
// which apply the penalties per sequence.

// ============================================================================
// Device Helper Functions
// ============================================================================

// Build recent token bitset collaboratively
template <int THREADS>
__device__ void build_recent_bitset(
    uint32_t* __restrict__ bitset,
    const int32_t* __restrict__ recent_tokens,
    int recent_len,
    int bitset_words
) {
    const int tid = threadIdx.x;
    
    // Clear bitset
    for (int i = tid; i < bitset_words; i += THREADS) {
        bitset[i] = 0;
    }
    __syncthreads();
    
    // Guard against null pointer or empty list
    if (recent_tokens == nullptr || recent_len <= 0) {
        return;  // Bitset already cleared, nothing more to do
    }
    
    // Set bits for recent tokens
    for (int i = tid; i < recent_len; i += THREADS) {
        int token = recent_tokens[i];
        if (token >= 0 && token < bitset_words * 32) {
            atomicOr(&bitset[token >> 5], 1u << (token & 31));
        }
    }
    __syncthreads();
}

// O(1) lookup for recent token
__device__ __forceinline__ bool is_in_recent(
    const uint32_t* __restrict__ bitset,
    int token_id
) {
    return (bitset[token_id >> 5] >> (token_id & 31)) & 1;
}

// ============================================================================
// DRY (Don't Repeat Yourself) Penalty - OPTIMIZED O(n² + V) Algorithm
// ============================================================================
// DRY penalty detects when the model is about to continue a repeated n-gram.
// Given recent tokens [t0, t1, ..., tn] and candidate token t:
// 1. Look at the suffix [t_{n-k}, ..., tn] for various lengths k
// 2. Find if this suffix appeared earlier in the context
// 3. If the token after that earlier occurrence is t, penalize based on match length
//
// Example: Context "the cat sat on the cat" + candidate "sat"
//   Suffix "the cat" appeared before, and was followed by "sat"
//   So "sat" gets penalized with match_length=2
//
// OPTIMIZATION: Instead of computing O(n²) per vocab token (total O(V×n²)),
// we precompute penalties for all positions ONCE (O(n²)), then look up in O(1).
// Total complexity: O(n² + V) instead of O(V × n²)

// Note: DRY_CACHE_MAX_ENTRIES is defined earlier (line ~168) = 512

// Structure to hold precomputed DRY penalties
// Stored in shared memory for fast lookup
// Uses DRY_CACHE_MAX_ENTRIES defined at top of file
struct DryPenaltyCache {
    int32_t token_ids[DRY_CACHE_MAX_ENTRIES];   // Token IDs that have penalties
    float penalties[DRY_CACHE_MAX_ENTRIES];      // Corresponding penalties
    int num_entries;                              // Number of valid entries
};

// Per-token flags DRY reads (`dry_reference.rs` builds them from the vocabulary).
constexpr uint8_t DRY_REFERENCE = 1;  // the token's text marks a reference
constexpr uint8_t DRY_JOINS = 2;      // the token carries on the word before it

// How far either side of a token the word around it is walked.
constexpr int DRY_WORD_REACH = 32;

// Whether `tokens[p]` sits inside a reference: any token of the word around it
// — the run of tokens joined without whitespace — is a reference token. A
// number, an id or a path is only right when it repeats exactly, so DRY never
// pushes one off its second spelling.
__device__ __forceinline__ bool dry_in_reference(
    const int32_t* __restrict__ tokens,
    int len,
    int p,
    const uint8_t* __restrict__ ref_flags
) {
    int start = p;
    for (int k = 0; k < DRY_WORD_REACH && start > 0 && (ref_flags[tokens[start]] & DRY_JOINS); k++) {
        start--;
    }
    int end = p;
    for (int k = 0; k < DRY_WORD_REACH && end + 1 < len && (ref_flags[tokens[end + 1]] & DRY_JOINS); k++) {
        end++;
    }
    for (int i = start; i <= end; i++) {
        if (ref_flags[tokens[i]] & DRY_REFERENCE) {
            return true;
        }
    }
    return false;
}

// Precompute DRY penalties for a single sequence
// Called ONCE per batch item, O(n²) complexity
// Results stored in shared memory cache for O(1) lookup
template <int THREADS>
__device__ void precompute_dry_penalties(
    const int32_t* __restrict__ recent_tokens,  // This sequence's recent tokens
    int recent_len,                              // Actual length of recent tokens
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range,                               // How far back to look (0 = full range)
    const uint8_t* __restrict__ ref_flags,       // [vocab] DRY_REFERENCE/DRY_JOINS, or null
    DryPenaltyCache* cache                       // Output: shared memory cache
) {
    const int tid = threadIdx.x;
    
    // Thread 0 initializes the cache
    if (tid == 0) {
        cache->num_entries = 0;
    }
    __syncthreads();
    
    // Early exit if DRY disabled or not enough context
    if (dry_multiplier == 0.0f || recent_len < 2) {
        return;
    }
    
    // Determine search range
    int search_end = recent_len - 1;  // Last position in suffix
    int search_start = (dry_range > 0 && dry_range < recent_len) ? (recent_len - dry_range) : 0;
    int num_positions = search_end - search_start;
    
    // Temporary storage for this thread's findings
    // We'll use a small local buffer then merge
    constexpr int LOCAL_BUFFER_SIZE = 16;
    int32_t local_tokens[LOCAL_BUFFER_SIZE];
    float local_penalties[LOCAL_BUFFER_SIZE];
    int local_count = 0;
    
    // Each thread processes a subset of positions
    for (int idx = tid; idx < num_positions; idx += THREADS) {
        int match_pos = search_start + idx;
        
        // Find longest suffix match ending at match_pos
        int match_len = 0;
        for (int offset = 1; offset <= min(match_pos + 1, recent_len); offset++) {
            int suffix_idx = recent_len - offset;      // Index in suffix (from end)
            int match_idx = match_pos - offset + 1;    // Corresponding position earlier
            
            if (match_idx < search_start) break;       // Don't go before search range
            if (recent_tokens[suffix_idx] != recent_tokens[match_idx]) break;
            
            match_len = offset;
        }
        
        // If match exceeds allowed length, record penalty for next token —
        // unless that token is part of a reference, which repeats by design.
        if (match_len > dry_allowed_length && match_pos + 1 < recent_len &&
            !(ref_flags != nullptr &&
              dry_in_reference(recent_tokens, recent_len, match_pos + 1, ref_flags))) {
            int32_t next_token = recent_tokens[match_pos + 1];
            int excess = match_len - dry_allowed_length;
            float penalty = dry_multiplier * powf(dry_base, (float)excess);
            
            // Store locally first
            if (local_count < LOCAL_BUFFER_SIZE) {
                local_tokens[local_count] = next_token;
                local_penalties[local_count] = penalty;
                local_count++;
            }
        }
    }
    
    __syncthreads();
    
    // Merge local buffers into shared cache using atomic operations
    // This handles deduplication by taking max penalty per token
    for (int i = 0; i < local_count; i++) {
        int32_t token = local_tokens[i];
        float penalty = local_penalties[i];
        
        // Check if token already exists in cache
        bool found = false;
        for (int j = 0; j < cache->num_entries && !found; j++) {
            if (cache->token_ids[j] == token) {
                // Use atomic max to avoid race condition when multiple threads
                // try to update the same token's penalty simultaneously
                atomicMaxFloat(&cache->penalties[j], penalty);
                found = true;
            }
        }
        
        // Add new entry if not found
        if (!found) {
            int slot = atomicAdd(&cache->num_entries, 1);
            if (slot < DRY_CACHE_MAX_ENTRIES) {
                cache->token_ids[slot] = token;
                cache->penalties[slot] = penalty;
            }
        }
    }
    
    __syncthreads();
    
    // Final deduplication pass - merge duplicate entries (taking max)
    // Use O(n log n) approach: insertion sort by token_id, then linear merge
    // Only thread 0 does this to avoid races
    if (tid == 0 && cache->num_entries > 1) {
        int n = cache->num_entries;
        
        // Clamp to valid range
        if (n > DRY_CACHE_MAX_ENTRIES) {
            n = DRY_CACHE_MAX_ENTRIES;
        }
        
        // Step 1: Insertion sort by token_id - O(n²) but fast for small n (~100)
        // and much better cache behavior than the nested loop dedup
        for (int i = 1; i < n; i++) {
            int32_t key_tok = cache->token_ids[i];
            float key_pen = cache->penalties[i];
            int j = i - 1;
            
            while (j >= 0 && cache->token_ids[j] > key_tok) {
                cache->token_ids[j + 1] = cache->token_ids[j];
                cache->penalties[j + 1] = cache->penalties[j];
                j--;
            }
            cache->token_ids[j + 1] = key_tok;
            cache->penalties[j + 1] = key_pen;
        }
        
        // Step 2: Linear merge of adjacent duplicates - O(n)
        int write_idx = 0;
        for (int i = 1; i < n; i++) {
            if (cache->token_ids[i] == cache->token_ids[write_idx]) {
                // Merge: take max penalty
                cache->penalties[write_idx] = fmaxf(cache->penalties[write_idx], cache->penalties[i]);
            } else {
                // New unique token
                write_idx++;
                if (write_idx != i) {
                    cache->token_ids[write_idx] = cache->token_ids[i];
                    cache->penalties[write_idx] = cache->penalties[i];
                }
            }
        }
        cache->num_entries = write_idx + 1;
    }
    
    __syncthreads();
}

// Branchless lookup in precomputed DRY cache - avoids warp divergence
// All threads execute the same number of iterations regardless of match position.
// Cache is in shared memory so extra reads are cheap (typically < 100 entries).
__device__ __forceinline__ float lookup_dry_penalty(
    int32_t token_id,
    const DryPenaltyCache* cache
) {
    float penalty = 0.0f;
    for (int i = 0; i < cache->num_entries; i++) {
        // Branchless select: compiles to selp instruction (no warp divergence)
        int match = (cache->token_ids[i] == token_id);
        penalty = match ? cache->penalties[i] : penalty;
    }
    return penalty;
}

// Legacy function for backward compatibility (not used in optimized path)
// Compute DRY penalty for a single token - O(n²) per call, DO NOT USE in hot path
__device__ __forceinline__ float compute_dry_penalty(
    int32_t candidate_token,
    const int32_t* __restrict__ recent_tokens,  // This sequence's recent tokens
    int recent_len,                              // Actual length of recent tokens
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range                               // How far back to look (0 = full range)
) {
    // Early exit if DRY disabled or not enough context
    if (dry_multiplier == 0.0f || recent_len < 2) {
        return 0.0f;
    }
    
    // Determine search range
    int search_end = recent_len - 1;  // Last position in suffix
    int search_start = (dry_range > 0 && dry_range < recent_len) ? (recent_len - dry_range) : 0;
    
    // Find the longest match
    int max_match_length = 0;
    
    // For each potential match position (excluding the current suffix)
    for (int match_pos = search_start; match_pos < search_end; match_pos++) {
        // Let's find how long a suffix match we can get ending at match_pos
        int match_len = 0;
        for (int offset = 1; offset <= min(match_pos + 1, recent_len); offset++) {
            int suffix_idx = recent_len - offset;      // Index in suffix (from end)
            int match_idx = match_pos - offset + 1;    // Corresponding position earlier
            
            if (match_idx < search_start) break;       // Don't go before search range
            if (recent_tokens[suffix_idx] != recent_tokens[match_idx]) break;
            
            match_len = offset;
        }
        
        // If we found a match of length >= 1, check if the next token matches candidate
        if (match_len >= 1 && match_pos + 1 < recent_len) {
            if (recent_tokens[match_pos + 1] == candidate_token) {
                max_match_length = max(max_match_length, match_len);
            }
        }
    }
    
    // Apply penalty if match exceeds allowed length
    if (max_match_length > dry_allowed_length) {
        int excess = max_match_length - dry_allowed_length;
        return dry_multiplier * powf(dry_base, (float)excess);
    }
    
    return 0.0f;
}

// Truly branchless penalty application (per-sequence aware)
__device__ __forceinline__ float apply_penalties_branchless(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset
) {
    // Protected tokens: EOS and the segment-close token must never be penalised.
    // Penalties would suppress the model's ability to stop generating.
    // Save the logit so we can restore it for protected tokens after penalties.
    float saved_logit = logit;
    int eos_diff = token_id - params.eos_token_id;
    int eos_abs  = (eos_diff ^ (eos_diff >> 31)) - (eos_diff >> 31);
    int is_eos   = 1 - min(eos_abs, 1);
    int segment_close_diff = token_id - params.segment_close_token_id;
    int segment_close_abs  = (segment_close_diff ^ (segment_close_diff >> 31)) - (segment_close_diff >> 31);
    int is_segment_close   = 1 - min(segment_close_abs, 1);
    int is_protected = is_eos | is_segment_close;

    // 1. Repeat Penalty (truly branchless, per-sequence via recent_bitset)
    // Defensive check: token_id should always be in [0, vocab_size) but guard anyway
    uint32_t in_recent = 0;
    if (token_id >= 0) {
        in_recent = (recent_bitset[token_id >> 5] >> (token_id & 31)) & 1;
    }
    
    // Sign-based penalty: positive logits divided, negative multiplied
    uint32_t logit_bits = __float_as_uint(logit);
    uint32_t sign_bit = logit_bits >> 31;  // 0 for positive, 1 for negative
    
    float inv_penalty = __fdividef(1.0f, params.repeat_penalty);
    float sign_f = __int2float_rn(sign_bit);
    float repeat_factor = sign_f * params.repeat_penalty + (1.0f - sign_f) * inv_penalty;
    
    float in_recent_f = __int2float_rn(in_recent);
    float final_factor = in_recent_f * repeat_factor + (1.0f - in_recent_f) * 1.0f;
    logit *= final_factor;
    
    // 2. Frequency Penalty (per-sequence token counts)
    // NOTE: token_counts MUST be [batch_size, vocab_size] layout (replicated if needed)
    // The caller is responsible for ensuring this - see Rust wrapper which replicates
    // a shared [vocab_size] array to per-sequence layout.
    int32_t count = 0;
    if (params.token_counts != nullptr && params.vocab_size > 0 && token_id < params.vocab_size) {
        // Per-sequence indexing: token_counts[batch_idx * vocab_size + token_id]
        count = __ldg(&params.token_counts[batch_idx * params.vocab_size + token_id]);
    }
    logit -= __int2float_rn(count) * params.frequency_penalty;
    
    // 3. Presence Penalty (per-sequence, based on same token_counts)
    int present = min(count, 1);
    logit -= __int2float_rn(present) * params.presence_penalty;

    // 3b. Cross-turn Penalty (tokens seen in prior turns, lighter penalty)
    if (params.cross_turn_counts != nullptr && params.cross_turn_penalty != 0.0f
        && params.vocab_size > 0 && token_id < params.vocab_size) {
        int32_t cross_count = __ldg(&params.cross_turn_counts[batch_idx * params.vocab_size + token_id]);
        int cross_present = min(cross_count, 1);
        logit -= __int2float_rn(cross_present) * params.cross_turn_penalty;
    }

    // Restore logit for protected tokens (EOS, segment-close) — undo all penalties above
    float pf = __int2float_rn(is_protected);
    logit = fmaf(pf, saved_logit - logit, logit);
    
    // 4. EOS Boost (truly branchless, with optional linear ramp)
    //    Reuses is_eos computed above for protection.
    //    Ramp: effective = eos_boost * clamp((current_len - ramp_start) / (ramp_len - ramp_start), 0, 1) * max_mult
    //    When ramp disabled (ramp_len <= 0 or max_mult <= 0): effective = eos_boost (flat)
    {
        float eos_f = __int2float_rn(is_eos);
        int ramp_span = max(params.eos_ramp_len - params.eos_ramp_start, 1);
        float t = fminf(__fdividef(float(max(params.current_len - params.eos_ramp_start, 0)),
                                   float(ramp_span)), 1.0f);
        int use_ramp = (params.eos_ramp_len > 0) & (int)(params.eos_boost_max_multiplier > 0.0f);
        float use_ramp_f = __int2float_rn(use_ramp);
        // When ramp active: boost * t * max_mult.  When inactive: boost * 1.0
        float multiplier = fmaf(use_ramp_f, t * params.eos_boost_max_multiplier - 1.0f, 1.0f);
        logit = fmaf(eos_f, params.eos_boost * multiplier, logit);
    }

    // 4b. Per-segment close boost (truly branchless, same ramp formula). Ramps on
    //     segment_len — which resets each segment — so it fires in EVERY segment,
    //     early ones included.  Targets BOTH the segment-close token AND EOS: inside
    //     a steered segment the model's EOS is intercepted into a segment close, so
    //     EOS is a second per-segment close lever and must get the same per-segment
    //     ramp, not just the total-length EOS boost above (which is dormant until
    //     late in the turn).  Outside a segment (segment_len == 0) this contributes
    //     nothing, so EOS there is governed solely by the total-length boost/failsafe.
    {
        float segment_close_f = __int2float_rn(is_segment_close | is_eos);
        int ramp_span = max(params.segment_close_ramp_len - params.segment_close_ramp_start, 1);
        float t = fminf(__fdividef(float(max(params.segment_len - params.segment_close_ramp_start, 0)),
                                   float(ramp_span)), 1.0f);
        int use_ramp = (params.segment_close_ramp_len > 0) & (int)(params.segment_close_max_multiplier > 0.0f);
        float use_ramp_f = __int2float_rn(use_ramp);
        float multiplier = fmaf(use_ramp_f, t * params.segment_close_max_multiplier - 1.0f, 1.0f);
        logit = fmaf(segment_close_f, params.segment_close_boost * multiplier, logit);
    }
    
    // 5. Banned Tokens (per-sequence or shared)
    float banned_mult = 1.0f;
    if (params.banned_tokens != nullptr) {  // Guard against null pointer
        if (params.banned_tokens_per_seq > 0) {
            // Per-sequence banned tokens
            const int32_t* my_banned = params.banned_tokens + batch_idx * params.banned_tokens_per_seq;
            for (int i = 0; i < params.banned_tokens_per_seq; i++) {
                int banned_token = my_banned[i];
                // Use -1 as sentinel for "no more banned tokens"
                int is_valid = (banned_token >= 0);
                int is_match = is_valid & (banned_token == token_id);
                banned_mult *= (1.0f - __int2float_rn(is_match));
            }
        } else if (params.num_banned_tokens > 0) {
            // Shared banned tokens (legacy behavior)
            for (int i = 0; i < params.num_banned_tokens; i++) {
                int is_match = (params.banned_tokens[i] == token_id);
                banned_mult *= (1.0f - __int2float_rn(is_match));
            }
        }
    }
    // If banned, set to -inf (branchless via selp)
    // Note: Cannot use `banned_mult * logit - (1-banned_mult) * INFINITY` because
    // when banned_mult==1.0 (no ban), 0.0f * INFINITY = NaN in IEEE 754.
    logit = (banned_mult < 1.0f) ? -INFINITY : logit;

    // 5b. The row's allowed-token bitmap: one read per token, however many the
    //     grammar forbids.
    if (params.allowed_bits != nullptr) {
        const uint32_t word = params.allowed_bits[token_id >> 5];
        logit = ((word >> (token_id & 31)) & 1u) ? logit : -INFINITY;
    }

    // 6. Token suppression (in-segment ceiling lever).
    // `suppress_penalty` is non-zero only while this sequence is inside a segment
    // (the caller zeroes it otherwise), so tokens outside the segment are never
    // touched.  Subtract it once from every token in the shared suppress list.
    if (params.suppress_penalty != 0.0f && params.suppress_tokens != nullptr) {
        float suppress_mult = 0.0f;
        for (int i = 0; i < params.suppress_count; i++) {
            int t = params.suppress_tokens[i];
            int is_match = (t >= 0) & (t == token_id);
            suppress_mult += __int2float_rn(is_match);
        }
        logit -= suppress_mult * params.suppress_penalty;
    }

    return logit;
}

// Extended penalty application with OPTIMIZED DRY using precomputed cache
// Use this version when you have precomputed the DRY penalties - O(1) lookup
__device__ __forceinline__ float apply_penalties_with_dry_cached(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache  // Precomputed DRY penalties (shared memory)
) {
    // Apply standard penalties first
    logit = apply_penalties_branchless(logit, token_id, batch_idx, params, recent_bitset);
    
    // Apply DRY penalty from precomputed cache - O(1) average lookup
    if (dry_cache != nullptr && dry_cache->num_entries > 0) {
        float dry_penalty = lookup_dry_penalty(token_id, dry_cache);
        logit -= dry_penalty;
    }
    
    return logit;
}

// Legacy: Extended penalty application including DRY penalty (SLOW - O(n²) per token)
// DEPRECATED: Use precompute_dry_penalties + apply_penalties_with_dry_cached instead
__device__ __forceinline__ float apply_penalties_with_dry(
    float logit,
    int token_id,
    int batch_idx,
    const PenaltyParams& params,
    const uint32_t* __restrict__ recent_bitset,
    const int32_t* __restrict__ recent_tokens,  // [batch_size * max_recent_len]
    const int32_t* __restrict__ recent_lens,    // [batch_size]
    int max_recent_len,
    float dry_multiplier,
    float dry_base,
    int dry_allowed_length,
    int dry_range
) {
    // Apply standard penalties first
    logit = apply_penalties_branchless(logit, token_id, batch_idx, params, recent_bitset);
    
    // Apply DRY penalty if enabled and we have recent tokens
    if (dry_multiplier != 0.0f && recent_tokens != nullptr && recent_lens != nullptr) {
        const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
        int my_recent_len = recent_lens[batch_idx];
        
        float dry_penalty = compute_dry_penalty(
            token_id, my_recent, my_recent_len,
            dry_multiplier, dry_base, dry_allowed_length, dry_range
        );
        logit -= dry_penalty;
    }
    
    return logit;
}
