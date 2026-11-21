// Vectorised max and softmax-sum passes, block reductions, and the tiled
// vocabulary pass.
//
// Holds the FP32 float4 max and softmax-sum passes, the block reductions, and
// the tile processors that load logits with penalties.
// Holds `tiled_sampling_pass`, which computes the max, softmax sum and top-k
// candidates in one scan of the vocabulary.

// ============================================================================
// Vectorized Pass Helpers (FP32 with float4 loads)
// ============================================================================

// Vectorized max-finding pass with penalties (FP32 only)
// Falls back to scalar loads when pointer is not 16-byte aligned
// (e.g., when vocab_size is not a multiple of 4 and batch_idx > 0).
template <int THREADS>
__device__ void find_max_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float& local_max,
    int& local_best
) {
    const int tid = threadIdx.x;
    local_max = -INFINITY;
    local_best = 0;
    
    const bool aligned = ((uintptr_t)logits % 16) == 0;
    
    if (aligned) {
        // Number of complete float4 groups
        const int vec_size = vocab_size / 4;
        const int remainder_start = vec_size * 4;
        
        // Vectorized pass: process 4 elements at a time
        const float4* logits_vec = reinterpret_cast<const float4*>(logits);
        
        #pragma unroll 2
        for (int i = tid; i < vec_size; i += THREADS) {
            float4 v = __ldg(&logits_vec[i]);
            int base_idx = i * 4;
            
            // Unroll 4 elements (fixed count)
            float logits_arr[4] = {v.x, v.y, v.z, v.w};
            
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = logits_arr[j];
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                int is_better = (int)(logit > local_max);
                local_max = fmaxf(local_max, logit);
                local_best = is_better * token_idx + (1 - is_better) * local_best;
            }
        }
        
        // Handle remainder (typically 0-3 elements, no unroll needed)
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_best = is_better * i + (1 - is_better) * local_best;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_best = is_better * i + (1 - is_better) * local_best;
        }
    }
}

// Vectorized softmax sum pass with penalties (FP32 only)
template <int THREADS>
__device__ float compute_softmax_sum_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float global_max,
    float inv_temp
) {
    const int tid = threadIdx.x;
    float local_sum = 0.f;
    
    const bool aligned = ((uintptr_t)logits % 16) == 0;
    
    if (aligned) {
        // Number of complete float4 groups
        const int vec_size = vocab_size / 4;
        const int remainder_start = vec_size * 4;
        
        // Vectorized pass
        const float4* logits_vec = reinterpret_cast<const float4*>(logits);
        
        #pragma unroll 2
        for (int i = tid; i < vec_size; i += THREADS) {
            float4 v = __ldg(&logits_vec[i]);
            int base_idx = i * 4;
            
            float logits_arr[4] = {v.x, v.y, v.z, v.w};
            
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = logits_arr[j];
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
                local_sum += prob;
            }
        }
        
        // Handle remainder
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    }
    
    return local_sum;
}

// ============================================================================
// Block Reduction Functions
// ============================================================================

// Block-level max reduction with index tracking
template <int THREADS>
__device__ void block_reduce_max_with_idx(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    float& out_max,
    int& out_idx
) {
    const int tid = threadIdx.x;
    
    // log2(THREADS) iterations of branchless max reduction
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(vals[tid + s] > vals[tid]);
            vals[tid] = fmaxf(vals[tid], vals[tid + s]);
            idxs[tid] = other_better * idxs[tid + s] + (1 - other_better) * idxs[tid];
        }
        __syncthreads();
    }
    
    out_max = vals[0];
    out_idx = idxs[0];
}

// Block-level sum reduction
template <int THREADS>
__device__ float block_reduce_sum(float val, float* __restrict__ smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    
    // log2(THREADS) iterations of sum reduction
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    return smem[0];
}

// ============================================================================
// Tiled Vocabulary Processing
// ============================================================================
// Process vocabulary in tiles to maximize shared memory reuse.
// Each tile: load logits → apply penalties → store in smem → reuse for:
//   1. Max-finding pass
//   2. Softmax sum pass
//   3. Top-k collection pass
// This reduces global memory reads from 3x to 1x.

// Load a tile of logits, apply penalties (including DRY), store to shared memory
template <typename T, int THREADS>
__device__ void load_tile_with_penalties(
    const T* __restrict__ logits,
    int tile_start,
    int tile_size,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties (nullptr if disabled)
    float* __restrict__ tile_buffer
) {
    const int tid = threadIdx.x;
    
    // Each thread loads multiple elements
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        float logit;
        
        if (global_idx < vocab_size) {
            logit = load_as_float(logits, global_idx);
            
            if (penalties != nullptr) {
                // Apply standard penalties + DRY from precomputed cache
                logit = apply_penalties_with_dry_cached(logit, global_idx, batch_idx, *penalties, recent_bitset, dry_cache);
            }
        } else {
            logit = -INFINITY;  // Padding for out-of-bounds
        }
        
        tile_buffer[i] = logit;
    }
    __syncthreads();
}

// Process tile for max-finding (returns local max and index)
template <int THREADS>
__device__ void process_tile_for_max(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float& local_max,
    int& local_idx
) {
    const int tid = threadIdx.x;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            int is_better = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_better * global_idx + (1 - is_better) * local_idx;
        }
    }
}

// Process tile for softmax sum
template <int THREADS>
__device__ float process_tile_for_softmax(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float global_max,
    float inv_temp
) {
    const int tid = threadIdx.x;
    float local_sum = 0.f;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            local_sum += prob;
        }
    }
    
    return local_sum;
}

// Process tile for top-k collection
template <int THREADS>
__device__ void process_tile_for_topk(
    const float* __restrict__ tile_buffer,
    int tile_start,
    int tile_size,
    int vocab_size,
    float threshold,
    float global_max,
    float inv_temp,
    float* __restrict__ out_probs,
    int* __restrict__ out_idxs,
    int max_k,
    int* __restrict__ count_ptr
) {
    const int tid = threadIdx.x;
    
    // TILE_SIZE/THREADS = 1 iteration per thread
    #pragma unroll 1
    for (int i = tid; i < tile_size; i += THREADS) {
        int global_idx = tile_start + i;
        if (global_idx < vocab_size) {
            float logit = tile_buffer[i];
            float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - global_max) * inv_temp);
            
            if (prob >= threshold) {
                int idx = atomicAdd(count_ptr, 1);
                if (idx < max_k) {
                    out_probs[idx] = prob;
                    out_idxs[idx] = global_idx;
                }
            }
        }
    }
}

// Complete tiled sampling pass: max + softmax + topk in one vocab scan
template <typename T, int THREADS, int MAX_K>
__device__ int tiled_sampling_pass(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    int top_k,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties
    float* __restrict__ tile_buffer,
    float* __restrict__ reduction_scratch,
    int* __restrict__ reduction_idx_scratch,
    float& out_global_max,
    float& out_global_sum,
    float* __restrict__ topk_probs,
    int* __restrict__ topk_idxs,
    int* __restrict__ topk_count,
    float inv_temp
) {
    const int tid = threadIdx.x;
    const int num_tiles = (vocab_size + TILE_SIZE - 1) / TILE_SIZE;
    
    // =========================================================================
    // SINGLE-PASS with Online Max Tracking (Approximate Max Algorithm)
    // =========================================================================
    // Instead of two passes (find max, then softmax), we use an online algorithm:
    // 1. Track running max as we process tiles
    // 2. Compute softmax contributions using current max estimate
    // 3. When max increases, apply correction factor to running sum
    //
    // Key insight: exp(x - new_max) = exp(x - old_max) * exp(old_max - new_max)
    // So if max increases by delta, multiply accumulated sum by exp(-delta)
    // =========================================================================
    
    // Thread-local state for online softmax
    float local_max = -INFINITY;
    float local_sum = 0.f;
    int local_max_idx = 0;
    
    // Reset top-k counter
    if (tid == 0) {
        *topk_count = 0;
    }
    __syncthreads();
    
    // Process all tiles in a single pass
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_size = min(TILE_SIZE, vocab_size - tile_start);
        
        // Load tile with penalties applied (including DRY from cache)
        load_tile_with_penalties<T, THREADS>(
            logits, tile_start, tile_size, vocab_size,
            batch_idx, penalties, recent_bitset, dry_cache, tile_buffer
        );
        
        // Process this tile: update max, accumulate softmax with correction
        // TILE_SIZE/THREADS = 1 iteration per thread
        #pragma unroll 1
        for (int i = tid; i < tile_size; i += THREADS) {
            int global_idx = tile_start + i;
            if (global_idx < vocab_size) {
                float logit = tile_buffer[i];
                
                // Online max update with correction
                if (logit > local_max) {
                    // New max found - correct accumulated sum
                    // sum_new = sum_old * exp(old_max - new_max)
                    float correction = fast_exp::exp<float, fast_exp::Softmax>((local_max - logit) * inv_temp);
                    local_sum *= correction;
                    local_max_idx = global_idx;
                    local_max = logit;
                }
                
                // Accumulate softmax contribution using current max
                float prob = fast_exp::exp<float, fast_exp::Softmax>((logit - local_max) * inv_temp);
                local_sum += prob;
            }
        }
        __syncthreads();
    }
    
    // =========================================================================
    // Block-wide reduction: combine local max/sum with online correction
    // =========================================================================
    reduction_scratch[tid] = local_max;
    reduction_idx_scratch[tid] = local_max_idx;
    __syncthreads();
    
    // Find block-wide max first - branchless (log2(THREADS) iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(reduction_scratch[tid + s] > reduction_scratch[tid]);
            reduction_scratch[tid] = fmaxf(reduction_scratch[tid], reduction_scratch[tid + s]);
            reduction_idx_scratch[tid] = other_better * reduction_idx_scratch[tid + s] +
                                         (1 - other_better) * reduction_idx_scratch[tid];
        }
        __syncthreads();
    }
    
    float global_max = reduction_scratch[0];
    out_global_max = global_max;
    __syncthreads();
    
    // Now correct each thread's local_sum to use the global_max
    // local_sum was computed with local_max, need to adjust
    float correction = fast_exp::exp<float, fast_exp::Softmax>((local_max - global_max) * inv_temp);
    float corrected_sum = local_sum * correction;
    
    // Reduce corrected sums
    out_global_sum = block_reduce_sum<THREADS>(corrected_sum, reduction_scratch);
    __syncthreads();
    
    // =========================================================================
    // Second mini-pass: collect top-k candidates (tiles already in L2 cache)
    // =========================================================================
    // We need the global_max for proper probability comparison
    // But tiles are now in L2 cache, so this is much faster than global memory
    //
    // NOTE: This tiled collection path is only used when top_k <= 0 (no top-k filtering).
    // When top_k > 0, we use the radix select path which provides unbiased O(n) selection.
    // The threshold-based collection here would be biased toward lower token IDs due to
    // the atomicAdd race, but that's acceptable when we're collecting all tokens anyway.
    
    float prob_threshold = 0.0f;  // Collect first MAX_K tokens (approximate when top_k=0)
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_size = min(TILE_SIZE, vocab_size - tile_start);
        
        // Reload from L2 cache (penalties recomputed but logits cached)
        load_tile_with_penalties<T, THREADS>(
            logits, tile_start, tile_size, vocab_size,
            batch_idx, penalties, recent_bitset, dry_cache, tile_buffer
        );
        
        // Collect top-k candidates
        process_tile_for_topk<THREADS>(
            tile_buffer, tile_start, tile_size, vocab_size,
            prob_threshold, global_max, inv_temp,
            topk_probs, topk_idxs, top_k, topk_count
        );
        __syncthreads();  // Ensure all atomic updates visible before next tile
    }
    __syncthreads();  // Final sync before reading topk_count
    
    return min(*topk_count, top_k);
}
