// The stencil shared-memory layout and the stencil sampling kernel for small
// constrained vocabularies.
//
// Holds `StencilSharedMem`, `MAX_STENCIL_SIZE`, and `stencil_sampling_kernel`,
// which samples over a small constrained vocabulary.
// Holds `dispatch_stencil_sampling`, which launches it.

// ============================================================================
// Stencil Sampling Kernel (Optimized for small constrained vocabularies)
// ============================================================================

// When a stencil (list of allowed token IDs) is provided and small (~100 tokens),
// this kernel is MUCH faster than iterating over the full vocabulary.
// Iterates directly over stencil indices: O(stencil_size) vs O(vocab_size).
// For stencil_size=100 vs vocab_size=128K, this is ~1000x less work!

// Shared memory for stencil kernel
template <int MAX_STENCIL, int THREADS>
struct StencilSharedMem {
    // Stencil token IDs loaded into shared memory for fast access
    int32_t stencil_tokens[MAX_STENCIL];
    // Corresponding logit values (penalized)
    float stencil_logits[MAX_STENCIL];
    // Reduction scratch
    float reduction_max[THREADS];
    int reduction_idx[THREADS];
    // Global scalars
    float global_max;
    float global_sum;
};

// Maximum stencil size supported (fits comfortably in shared memory)
constexpr int MAX_STENCIL_SIZE = 1024;

// Stencil sampling kernel - optimized for small constrained vocabularies
// Iterates ONLY over allowed tokens, not full vocab
template <typename T, int MAX_STENCIL, int THREADS, bool USE_PENALTIES>
__global__ void stencil_sampling_kernel(
    // Input logits
    const T* __restrict__ logits,           // [batch_size, vocab_size]
    int vocab_size,
    
    // Stencil (allowed tokens) - shared across batch for now
    const int32_t* __restrict__ stencil,    // [stencil_size]
    int stencil_size,
    
    // Penalty scalars (applied to stencil tokens only)
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float eos_boost,
    int32_t eos_token_id,
    int32_t segment_close_token_id,
    
    // Penalty GPU pointers
    const int32_t* __restrict__ token_counts,     // [batch_size, vocab_size] or null
    const int32_t* __restrict__ banned_tokens,    // [num_banned] or null (rarely used with stencil)
    int32_t num_banned_tokens,
    
    // Recent tokens for repeat penalty
    const int32_t* __restrict__ recent_tokens,    // [batch_size * max_recent_len] or null
    const int32_t* __restrict__ recent_lens,      // [batch_size] or null
    int32_t max_recent_len,
    
    // Sampling configuration
    float temperature,
    int top_k,
    float top_p,
    
    // Output
    uint32_t* __restrict__ output_tokens,    // [batch_size]
    // [batch_size] or null: 1 where a stencil token's logit is NaN or +inf
    // (`logit_not_finite`), else 0 — every row's word written.
    uint32_t* __restrict__ nonfinite_rows,

    // RNG
    uint64_t seed,
    uint64_t* __restrict__ rng_offsets       // [batch_size]
) {
    __shared__ StencilSharedMem<MAX_STENCIL, THREADS> smem;

    // Dynamic shared memory for recent token bitset (if penalties used)
    extern __shared__ uint32_t recent_bitset[];

    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Safety checks
    if (output_tokens == nullptr || logits == nullptr || stencil == nullptr || stencil_size <= 0) {
        if (tid == 0 && output_tokens != nullptr) {
            output_tokens[batch_idx] = 0;
        }
        if (tid == 0 && nonfinite_rows != nullptr) {
            nonfinite_rows[batch_idx] = 0;
        }
        return;
    }
    
    // Clamp stencil size to supported maximum
    const int actual_stencil_size = min(stencil_size, MAX_STENCIL);
    
    const T* my_logits = logits + batch_idx * vocab_size;
    const int bitset_words = (vocab_size + 31) / 32;
    
    // =========================================================
    // STEP 1: Load stencil into shared memory (collaborative)
    // =========================================================
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        smem.stencil_tokens[i] = stencil[i];
    }
    __syncthreads();
    
    // =========================================================
    // STEP 2: Build recent token bitset (if penalties enabled)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (recent_tokens != nullptr && recent_lens != nullptr) {
            // Clear bitset
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            __syncthreads();
            
            // Get this sequence's recent tokens
            const int32_t* my_recent = recent_tokens + batch_idx * max_recent_len;
            int my_recent_len = recent_lens[batch_idx];
            
            // Set bits for recent tokens
            for (int i = tid; i < my_recent_len; i += THREADS) {
                int32_t tok = my_recent[i];
                if (tok >= 0 && tok < vocab_size) {
                    atomicOr(&recent_bitset[tok >> 5], 1u << (tok & 31));
                }
            }
            __syncthreads();
        } else {
            // Clear bitset if no recent tokens
            for (int i = tid; i < bitset_words; i += THREADS) {
                recent_bitset[i] = 0;
            }
            __syncthreads();
        }
    }
    
    // =========================================================
    // STEP 3: Load logits for stencil tokens, apply penalties
    // =========================================================
    bool bad = false;
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        int32_t token_id = smem.stencil_tokens[i];
        float logit = load_as_float(my_logits, token_id);
        bad |= logit_not_finite(logit);

        if constexpr (USE_PENALTIES) {
            // Protected tokens: EOS and the segment-close token must never be penalised
            float saved_logit = logit;
            int eos_d = token_id - eos_token_id;
            int eos_a = (eos_d ^ (eos_d >> 31)) - (eos_d >> 31);
            int is_eos = 1 - min(eos_a, 1);
            int segment_close_d = token_id - segment_close_token_id;
            int segment_close_a = (segment_close_d ^ (segment_close_d >> 31)) - (segment_close_d >> 31);
            int is_segment_close = 1 - min(segment_close_a, 1);
            int is_protected = is_eos | is_segment_close;

            // Apply repeat penalty (branchless - same pattern as apply_penalties_branchless)
            if (recent_tokens != nullptr) {
                uint32_t word = recent_bitset[token_id >> 5];
                uint32_t in_recent = (word >> (token_id & 31)) & 1u;
                float in_recent_f = __int2float_rn(in_recent);
                
                // Sign-based penalty: positive logits divided, negative multiplied
                uint32_t logit_bits = __float_as_uint(logit);
                uint32_t sign_bit = logit_bits >> 31;
                float sign_f = __int2float_rn(sign_bit);
                float inv_penalty = __fdividef(1.0f, repeat_penalty);
                float repeat_factor = sign_f * repeat_penalty + (1.0f - sign_f) * inv_penalty;
                float final_factor = in_recent_f * repeat_factor + (1.0f - in_recent_f) * 1.0f;
                logit *= final_factor;
            }
            
            // Apply frequency/presence penalties (branchless)
            if (token_counts != nullptr) {
                int32_t count = __ldg(&token_counts[batch_idx * vocab_size + token_id]);
                logit -= frequency_penalty * __int2float_rn(count);
                int present = min(count, 1);
                logit -= __int2float_rn(present) * presence_penalty;
            }

            // Restore logit for protected tokens (EOS, segment-close) — undo all penalties
            float pf = __int2float_rn(is_protected);
            logit = fmaf(pf, saved_logit - logit, logit);
            
            // Apply EOS boost (branchless)
            {
                logit += float(is_eos) * eos_boost;
            }
        }
        
        smem.stencil_logits[i] = logit;
    }
    const bool not_finite = __syncthreads_or(bad) != 0;
    if (tid == 0 && nonfinite_rows != nullptr) {
        nonfinite_rows[batch_idx] = not_finite ? 1u : 0u;
    }

    // =========================================================
    // STEP 4: Check for banned tokens (set to -inf)
    // =========================================================
    if constexpr (USE_PENALTIES) {
        if (banned_tokens != nullptr && num_banned_tokens > 0) {
            // For each banned token, check if it's in stencil and mark as -inf
            for (int b = tid; b < num_banned_tokens; b += THREADS) {
                int32_t banned_id = banned_tokens[b];
                // Linear search in stencil (fast for small stencils)
                for (int i = 0; i < actual_stencil_size; i++) {
                    if (smem.stencil_tokens[i] == banned_id) {
                        smem.stencil_logits[i] = -INFINITY;
                        break;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // =========================================================
    // STEP 5: Find max logit (for argmax or softmax normalization)
    // =========================================================
    float local_max = -INFINITY;
    int local_best_idx = 0;
    
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        float logit = smem.stencil_logits[i];
        int is_better = (int)(logit > local_max);
        local_max = fmaxf(local_max, logit);
        local_best_idx = is_better * i + (1 - is_better) * local_best_idx;
    }
    
    smem.reduction_max[tid] = local_max;
    smem.reduction_idx[tid] = local_best_idx;
    __syncthreads();
    
    // Block reduction for max (branchless)
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int other_better = (int)(smem.reduction_max[tid + stride] > smem.reduction_max[tid]);
            smem.reduction_max[tid] = fmaxf(smem.reduction_max[tid], smem.reduction_max[tid + stride]);
            smem.reduction_idx[tid] = other_better * smem.reduction_idx[tid + stride] +
                                      (1 - other_better) * smem.reduction_idx[tid];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        smem.global_max = smem.reduction_max[0];
    }
    __syncthreads();
    
    float global_max = smem.global_max;
    int best_stencil_idx = smem.reduction_idx[0];
    
    // =========================================================
    // STEP 6: Argmax path (temperature <= 0)
    // =========================================================
    if (!(temperature > 0.f)) {
        if (tid == 0) {
            output_tokens[batch_idx] = smem.stencil_tokens[best_stencil_idx];
        }
        return;
    }
    
    // =========================================================
    // STEP 7: Compute softmax probabilities
    // =========================================================
    const float inv_temp = 1.0f / temperature;
    float local_sum = 0.f;
    
    // Compute exp((logit - max) / temp) for each stencil token
    for (int i = tid; i < actual_stencil_size; i += THREADS) {
        float prob = fast_exp::exp<float, fast_exp::Softmax>((smem.stencil_logits[i] - global_max) * inv_temp);
        smem.stencil_logits[i] = prob;  // Reuse array for probabilities
        local_sum += prob;
    }
    __syncthreads();
    
    // Block reduction for sum
    smem.reduction_max[tid] = local_sum;  // Reuse for sum reduction
    __syncthreads();
    
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem.reduction_max[tid] += smem.reduction_max[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        smem.global_sum = smem.reduction_max[0];
    }
    __syncthreads();
    
    float global_sum = smem.global_sum;
    
    // Guard against zero sum
    if (global_sum <= 0.f) {
        if (tid == 0) {
            output_tokens[batch_idx] = smem.stencil_tokens[best_stencil_idx];
        }
        return;
    }
    
    // =========================================================
    // STEP 8: Apply top-k and top-p filtering (optional)
    // =========================================================
    int sample_size = actual_stencil_size;
    const float inv_sum = 1.0f / global_sum;
    
    // For small stencils, top-k/top-p typically don't apply much
    // but we support them for correctness
    if (top_k > 0 && top_k < actual_stencil_size) {
        // Use insertion sort for better cache behavior than bubble sort
        // O(n²) worst case but O(n) for nearly-sorted data, and better constants
        if (tid == 0) {
            // Insertion sort descending by probability
            for (int i = 1; i < actual_stencil_size; i++) {
                float key_prob = smem.stencil_logits[i];
                int32_t key_tok = smem.stencil_tokens[i];
                int j = i - 1;
                
                // Move elements smaller than key_prob to the right
                while (j >= 0 && smem.stencil_logits[j] < key_prob) {
                    smem.stencil_logits[j + 1] = smem.stencil_logits[j];
                    smem.stencil_tokens[j + 1] = smem.stencil_tokens[j];
                    j--;
                }
                smem.stencil_logits[j + 1] = key_prob;
                smem.stencil_tokens[j + 1] = key_tok;
            }
            sample_size = min(top_k, actual_stencil_size);
        }
        __syncthreads();
    }
    
    // Apply top-p (nucleus) filtering
    if (top_p < 1.0f && top_p > 0.f) {
        if (tid == 0) {
            float cumsum = 0.f;
            for (int i = 0; i < sample_size; i++) {
                cumsum += smem.stencil_logits[i] * inv_sum;
                if (cumsum >= top_p) {
                    sample_size = i + 1;
                    break;
                }
            }
        }
        __syncthreads();
    }
    
    // =========================================================
    // STEP 9: Sample from filtered distribution
    // =========================================================
    if (tid == 0) {
        uint64_t offset = rng_offsets ? rng_offsets[batch_idx] : 0;
        if (rng_offsets) {
            rng_offsets[batch_idx] = offset + 1;
        }
        
        // Generate uniform random in [0, 1)
        uint4 rand = flash::philox(seed, 0, offset);
        float u = (rand.x & 0x00FFFFFF) / 16777216.f;
        
        // Renormalize over sample_size tokens
        float total = 0.f;
        for (int i = 0; i < sample_size; i++) {
            total += smem.stencil_logits[i];
        }
        
        if (total <= 0.f) {
            output_tokens[batch_idx] = smem.stencil_tokens[0];
            return;
        }
        
        float cumsum = 0.f;
        for (int i = 0; i < sample_size; i++) {
            cumsum += smem.stencil_logits[i] / total;
            if (u < cumsum) {
                output_tokens[batch_idx] = smem.stencil_tokens[i];
                return;
            }
        }
        output_tokens[batch_idx] = smem.stencil_tokens[sample_size - 1];
    }
}

// Stencil dispatch helper
template <typename T>
inline void dispatch_stencil_sampling(
    const T* logits,
    int batch_size,
    int vocab_size,
    // Stencil
    const int32_t* stencil,
    int stencil_size,
    // Penalty scalars
    float repeat_penalty,
    float frequency_penalty,
    float presence_penalty,
    float eos_boost,
    int32_t eos_token_id,
    int32_t segment_close_token_id,
    // Penalty GPU pointers
    const int32_t* token_counts,
    const int32_t* banned_tokens,
    int32_t num_banned_tokens,
    // Recent tokens
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
    // Dispatch flag
    bool use_penalties,
    // The stream the launch goes on — the caller's, never the null stream
    cudaStream_t stream
) {
    
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    // Dynamic shared memory for bitset (only if penalties used)
    const size_t bitset_bytes = use_penalties ? ((vocab_size + 31) / 32) * sizeof(uint32_t) : 0;
    
    if (use_penalties) {
        stencil_sampling_kernel<T, MAX_STENCIL_SIZE, THREADS_PER_BLOCK, true>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                stencil, stencil_size,
                repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
                token_counts, banned_tokens, num_banned_tokens,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets
            );
    } else {
        stencil_sampling_kernel<T, MAX_STENCIL_SIZE, THREADS_PER_BLOCK, false>
            <<<grid, block, bitset_bytes, stream>>>(
                logits, vocab_size,
                stencil, stencil_size,
                repeat_penalty, frequency_penalty, presence_penalty, eos_boost, eos_token_id, segment_close_token_id,
                token_counts, banned_tokens, num_banned_tokens,
                recent_tokens, recent_lens, max_recent_len,
                temperature, top_k, top_p,
                output_tokens, nonfinite_rows, seed, rng_offsets
            );
    }
}
