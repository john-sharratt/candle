// Radix threshold search, top-k collection, and sampling from the candidates.
//
// Holds `radix_select_logit_threshold`, which finds the top-k threshold on raw
// logits in O(n), and the fused collect-and-normalise pass that computes
// candidate probabilities in shared memory.
// Holds `sample_from_topk`, which draws with Philox RNG, and `typical_accepts`,
// the Medusa typical acceptance over that distribution.

// ============================================================================
// Optimized Radix Select on RAW LOGITS (no exp, no global_max needed)
// ============================================================================
// Since softmax is monotonic, the top-k logits ARE the top-k probabilities.
// This eliminates exp() from all 4 radix passes AND removes the need for
// global max/sum passes entirely, reducing the top-k path from 8 to 5 passes.

// 8-bit radix select operating directly on (penalized) logit values.
// exp() is completely eliminated from the hot loop. `not_finite` is set, in
// every thread, to whether any raw logit of the row is NaN or +inf
// (`logit_not_finite`) — tested on the first pass, which reads every one.
template <typename T, int THREADS>
__device__ float radix_select_logit_threshold(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    int k,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,
    int* __restrict__ histogram,
    float* __restrict__ smem_scratch,
    bool& not_finite
) {
    const int tid = threadIdx.x;

    uint32_t prefix = 0;
    int remaining_k = k;
    bool bad = false;
    __shared__ int radix_broadcast[2];
    // Track the maximum sortable value seen across all tokens (computed during
    // pass 0 where matches=true for all tokens).  Used to clamp the threshold
    // so we never return a value below (max_logit - DEAD_ZONE), which would
    // cause collect_and_locally_normalize to include garbage (-inf / -100)
    // tokens when top_k exceeds the number of "live" tokens in the vocab.
    __shared__ uint32_t smem_max_sortable;
    if (tid == 0) smem_max_sortable = 0u;
    __syncthreads();

    for (int pass = 0; pass < 4; pass++) {
        int shift = 24 - pass * RADIX_BITS;
        int prefix_shift = shift + RADIX_BITS;
        
        for (int i = tid; i < RADIX_SIZE; i += THREADS) {
            histogram[i] = 0;
        }
        __syncthreads();
        
        // Key optimization: operate on raw logits, no exp()!
        // float_to_sortable handles negatives correctly.
        #pragma unroll 2
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            if (pass == 0) {
                bad |= logit_not_finite(logit);
            }

            if (penalties != nullptr) {
                logit = apply_penalties_with_dry_cached(
                    logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
            }

            uint32_t sortable = float_to_sortable(logit);

            // On pass 0 every token matches; track the global maximum.
            if (pass == 0) {
                atomicMax(&smem_max_sortable, sortable);
            }

            bool matches = (prefix_shift >= 32) ||
                           ((sortable >> prefix_shift) == (prefix >> prefix_shift));

            if (matches) {
                int digit = (sortable >> shift) & RADIX_MASK;
                atomicAdd(&histogram[digit], 1);
            }
        }
        // `bad` is set on the first pass only, so every pass's vote is the same.
        not_finite = __syncthreads_or(bad) != 0;
        
        if (tid == 0) {
            int cumsum = 0;
            int selected_bucket = 0;
            
            for (int b = RADIX_SIZE - 1; b >= 0; b--) {
                int count = histogram[b];
                if (cumsum + count >= remaining_k) {
                    selected_bucket = b;
                    remaining_k -= cumsum;
                    break;
                }
                cumsum += count;
            }
            
            prefix |= (uint32_t(selected_bucket) << shift);
            radix_broadcast[0] = remaining_k;
            radix_broadcast[1] = selected_bucket;
        }
        __syncthreads();
        
        remaining_k = radix_broadcast[0];
        int selected = radix_broadcast[1];
        prefix = (prefix & ~(uint32_t(RADIX_MASK) << shift)) |
                 (uint32_t(selected) << shift);
        __syncthreads();
    }
    
    float threshold = sortable_to_float(prefix);

    // SAFETY: If top_k exceeds the number of "live" tokens (e.g. most tokens
    // are masked to -100 or -inf), the radix select will push the threshold
    // into garbage territory, causing collect_and_locally_normalize to admit
    // those masked tokens.  Clamp the threshold so it is never more than
    // DEAD_ZONE logit units below the maximum penalised logit.  The value
    // 50.0 is generous; real model logit gaps are rarely > 30.
    constexpr float DEAD_ZONE = 50.0f;
    float max_logit = sortable_to_float(smem_max_sortable);
    threshold = fmaxf(threshold, max_logit - DEAD_ZONE);

    return threshold;  // Returns a logit threshold, not a probability
}

// ============================================================================
// Fused Collect + Local Softmax (single pass, probabilities computed in smem)
// ============================================================================
// Collects top-k candidates by raw logit threshold, then computes softmax
// ONLY over the collected candidates. No global softmax sum needed.
// This is the standard approach used by vLLM and HuggingFace:
// sampling only needs relative probabilities among the top-k set.

template <typename T, int THREADS>
__device__ int collect_and_locally_normalize(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float logit_threshold,     // Raw logit threshold from radix select
    float inv_temp,            // 1.0 / temperature
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,
    float* __restrict__ out_vals,     // Output: probabilities (after local softmax)
    int* __restrict__ out_idxs,       // Output: token indices
    int max_k,
    int* __restrict__ count_ptr,
    float* __restrict__ smem_scratch  // [THREADS] for reduction
) {
    const int tid = threadIdx.x;

    // ---- Two-phase collection ----
    //
    // Phase 1: Collect tokens STRICTLY ABOVE the threshold.
    // These are guaranteed top-(k-1) candidates regardless of ties.
    //
    // Phase 2: Fill remaining slots with tokens EQUAL TO the threshold
    // (the boundary value). Ties at the boundary are resolved by
    // lowest token-index first (deterministic across runs).
    //
    // Why two phases instead of single >= pass:
    //   A single >= pass uses atomicAdd to race-fill max_k slots. When
    //   many tokens tie at the boundary (e.g., all vocab at same logit),
    //   the top token may lose the race and be excluded.  Two phases
    //   guarantee the strictly-better tokens fill first.

    if (tid == 0) *count_ptr = 0;
    __syncthreads();

    // Phase 1: strictly above threshold
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        if (penalties != nullptr) {
            logit = apply_penalties_with_dry_cached(
                logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
        }

        if (logit > logit_threshold) {
            int idx = atomicAdd(count_ptr, 1);
            if (idx < max_k) {
                out_vals[idx] = logit;
                out_idxs[idx] = i;
            }
        }
    }
    __syncthreads();

    // Phase 2: exactly at threshold — fill remaining slots (if any)
    int phase1_count = min(*count_ptr, max_k);
    if (phase1_count < max_k) {
        if (tid == 0) *count_ptr = phase1_count; // reset to phase-1 fill level
        __syncthreads();

        #pragma unroll 2
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            if (penalties != nullptr) {
                logit = apply_penalties_with_dry_cached(
                    logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
            }

            if (logit == logit_threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_vals[idx] = logit;
                    out_idxs[idx] = i;
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    int num_collected = min(*count_ptr, max_k);

    // ---- Find max among collected candidates ----
    float collected_max_local = -INFINITY;
    for (int i = tid; i < num_collected; i += THREADS) {
        collected_max_local = fmaxf(collected_max_local, out_vals[i]);
    }
    smem_scratch[tid] = collected_max_local;
    __syncthreads();

    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem_scratch[tid] = fmaxf(smem_scratch[tid], smem_scratch[tid + s]);
        __syncthreads();
    }
    float collected_max = smem_scratch[0];
    __syncthreads();

    // ---- Convert collected logits -> normalised probabilities in-place ----
    float local_sum = 0.f;
    for (int i = tid; i < num_collected; i += THREADS) {
        float prob = fast_exp::exp<float, fast_exp::Softmax>(
            (out_vals[i] - collected_max) * inv_temp);
        out_vals[i] = prob;
        local_sum += prob;
    }
    __syncthreads();

    float total_sum = block_reduce_sum<THREADS>(local_sum, smem_scratch);
    __syncthreads();

    if (total_sum > 0.f) {
        float inv_total = 1.f / total_sum;
        for (int i = tid; i < num_collected; i += THREADS) {
            out_vals[i] *= inv_total;
        }
        __syncthreads();
    }

    return num_collected;
}

// ============================================================================
// Legacy Top-K Collection (threshold-based)
// ============================================================================

// Collect top-k tokens above threshold (templated for input type)
template <typename T, int THREADS>
__device__ int collect_topk_tokens_typed(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float prob_threshold,
    float global_max,
    float inv_temp,
    float inv_sum,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    const int tid = threadIdx.x;
    
    // Initialize count
    if (tid == 0) {
        *count_ptr = 0;
    }
    __syncthreads();
    
    // Phase 1: Collect elements STRICTLY above threshold.
    // These are guaranteed top-k members and must not be displaced by
    // threshold-equal tokens racing via atomicAdd.
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        
        if (penalties != nullptr) {
            logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
        }
        
        float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp) * inv_sum;
        
        if (prob > prob_threshold) {
            int idx = atomicAdd(count_ptr, 1);
            if (idx < max_k) {
                out_probs[idx] = prob;
                out_idxs[idx] = i;
            }
        }
    }
    __syncthreads();
    
    // Phase 2: Fill remaining slots from elements at the threshold boundary.
    // These are interchangeable (same probability), so order doesn't matter.
    if (*count_ptr < max_k) {
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = load_as_float(logits, i);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp) * inv_sum;
            
            if (prob == prob_threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_probs[idx] = prob;
                    out_idxs[idx] = i;
                }
            }
        }
        __syncthreads();
    }
    
    return min(*count_ptr, max_k);
}

// FP32 specialization for backward compatibility
template <int THREADS>
__device__ int collect_topk_tokens(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    float prob_threshold,
    float global_max,
    float inv_temp,
    float inv_sum,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    return collect_topk_tokens_typed<float, THREADS>(
        logits, vocab_size, batch_idx, prob_threshold, global_max, inv_temp, inv_sum,
        penalties, recent_bitset, out_probs, out_idxs, max_k, count_ptr
    );
}

// Sample from top-k candidates using Philox RNG
__device__ __forceinline__ int sample_from_topk(
    const float* __restrict__ probs,
    const int* __restrict__ indices,
    int num_candidates,
    uint64_t seed,
    uint64_t offset
) {
    // Safety: must have at least 1 candidate
    if (num_candidates <= 0) {
        return 0;  // Return token 0 as fallback (should never happen)
    }
    
    // Fast path: single candidate
    if (num_candidates == 1) {
        return indices[0];
    }
    
    // Generate uniform random in [0, 1)
    uint4 rand = flash::philox(seed, 0, offset);
    float u = (rand.x & 0x00FFFFFF) / 16777216.f;
    
    // Renormalize probabilities
    float total = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        total += probs[i];
    }
    
    // Guard against zero total (all -inf logits)
    if (total <= 0.f) {
        return indices[0];  // Return first candidate
    }
    
    float inv_total = 1.f / total;
    
    // Sample via CDF
    float cumsum = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        cumsum += probs[i] * inv_total;
        if (u < cumsum) return indices[i];
    }
    return indices[num_candidates - 1];
}

// Medusa typical acceptance over the distribution `sample_from_topk` draws
// from: the first `num_candidates` entries, renormalised. True when `draft`'s
// probability exceeds `min(eps, delta * exp(-entropy))`. A draft outside the
// candidate set has probability zero and is never accepted.
__device__ __forceinline__ bool typical_accepts(
    const float* __restrict__ probs,
    const int* __restrict__ indices,
    int num_candidates,
    int draft,
    float eps,
    float delta
) {
    float total = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        total += probs[i];
    }
    if (!(total > 0.f)) {
        return false;
    }
    const float inv_total = 1.f / total;
    float entropy = 0.f;
    float p_draft = 0.f;
    for (int i = 0; i < num_candidates; i++) {
        const float p = probs[i] * inv_total;
        if (p > 0.f) {
            entropy -= p * logf(p);
        }
        if (indices[i] == draft) {
            p_draft = p;
        }
    }
    return p_draft > fminf(eps, delta * expf(-entropy));
}
