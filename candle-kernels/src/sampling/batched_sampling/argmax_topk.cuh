// Argmax and the warp and shared-memory sorts used for top-k selection.
//
// Holds the typed and vectorised argmax, the warp shuffle compare-swap, bitonic
// and top-k sorts for k up to 32, and the hybrid bitonic sort used above that.
// Also holds the `float_to_sortable` and `sortable_to_float` mappings that the
// radix select orders logits by.

// ============================================================================
// Argmax Functions
// ============================================================================

// Vectorized argmax for greedy decoding (templated for input type).
// `not_finite` is set, in every thread, to whether any raw logit of the row is
// NaN or +inf (`logit_not_finite`).
template <typename T, int THREADS>
__device__ int branchless_argmax_typed(
    const T* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    const DryPenaltyCache* dry_cache,  // Precomputed DRY penalties
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs,
    bool& not_finite
) {
    const int tid = threadIdx.x;

    float local_max = -INFINITY;
    int local_idx = 0;
    bool bad = false;

    // Process logits with penalties (including DRY from cache)
    #pragma unroll 2
    for (int i = tid; i < vocab_size; i += THREADS) {
        float logit = load_as_float(logits, i);
        bad |= logit_not_finite(logit);

        if (penalties != nullptr) {
            logit = apply_penalties_with_dry_cached(logit, i, batch_idx, *penalties, recent_bitset, dry_cache);
        }

        int is_greater = (int)(logit > local_max);
        local_max = fmaxf(local_max, logit);
        local_idx = is_greater * i + (1 - is_greater) * local_idx;
    }

    // Store for reduction
    smem_vals[tid] = local_max;
    smem_idxs[tid] = local_idx;
    not_finite = __syncthreads_or(bad) != 0;
    
    // Tree reduction (log2(THREADS) = 8 iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(smem_vals[tid + s] > smem_vals[tid]);
            smem_vals[tid] = fmaxf(smem_vals[tid], smem_vals[tid + s]);
            smem_idxs[tid] = other_better * smem_idxs[tid + s] + 
                             (1 - other_better) * smem_idxs[tid];
        }
        __syncthreads();
    }
    
    return smem_idxs[0];
}

// Vectorized FP32 argmax with float4 loads (4x memory bandwidth)
// Falls back to scalar loads when pointer is not 16-byte aligned.
template <int THREADS>
__device__ int branchless_argmax_vectorized(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs
) {
    const int tid = threadIdx.x;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
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
            
            // Process each element of float4
            #pragma unroll 4
            for (int j = 0; j < 4; j++) {
                float logit = (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
                int token_idx = base_idx + j;
                
                if (penalties != nullptr) {
                    logit = apply_penalties_branchless(logit, token_idx, batch_idx, *penalties, recent_bitset);
                }
                
                int is_greater = (int)(logit > local_max);
                local_max = fmaxf(local_max, logit);
                local_idx = is_greater * token_idx + (1 - is_greater) * local_idx;
            }
        }
        
        // Handle remainder (vocab_size not divisible by 4)
        for (int i = remainder_start + tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_greater = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_greater * i + (1 - is_greater) * local_idx;
        }
    } else {
        // Scalar fallback for misaligned pointers
        for (int i = tid; i < vocab_size; i += THREADS) {
            float logit = __ldg(&logits[i]);
            
            if (penalties != nullptr) {
                logit = apply_penalties_branchless(logit, i, batch_idx, *penalties, recent_bitset);
            }
            
            int is_greater = (int)(logit > local_max);
            local_max = fmaxf(local_max, logit);
            local_idx = is_greater * i + (1 - is_greater) * local_idx;
        }
    }
    
    // Store for reduction
    smem_vals[tid] = local_max;
    smem_idxs[tid] = local_idx;
    __syncthreads();
    
    // Tree reduction (log2(THREADS) = 8 iterations)
    #pragma unroll
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int other_better = (int)(smem_vals[tid + s] > smem_vals[tid]);
            smem_vals[tid] = fmaxf(smem_vals[tid], smem_vals[tid + s]);
            smem_idxs[tid] = other_better * smem_idxs[tid + s] + 
                             (1 - other_better) * smem_idxs[tid];
        }
        __syncthreads();
    }
    
    return smem_idxs[0];
}

// FP32 specialization - uses vectorized version
template <int THREADS>
__device__ int branchless_argmax(
    const float* __restrict__ logits,
    int vocab_size,
    int batch_idx,
    const PenaltyParams* penalties,
    const uint32_t* __restrict__ recent_bitset,
    float* __restrict__ smem_vals,
    int* __restrict__ smem_idxs
) {
    return branchless_argmax_vectorized<THREADS>(
        logits, vocab_size, batch_idx, penalties, recent_bitset, smem_vals, smem_idxs
    );
}

// ============================================================================
// Warp Shuffle-Based Sorting for Small K (k ≤ 32)
// ============================================================================
// Uses warp shuffles instead of shared memory for much faster sorting.
// Each thread in a warp holds one (val, idx) pair and exchanges via shuffles.
// No __syncthreads() needed - warp-synchronous execution.

constexpr unsigned FULL_WARP_MASK = 0xFFFFFFFF;

// Warp-level compare-and-swap using shuffles
// For descending sort: keep larger value in lower lane when dir=true
__device__ __forceinline__ void warp_compare_swap(
    float& val, int& idx, int lane_mask, bool dir
) {
    float other_val = __shfl_xor_sync(FULL_WARP_MASK, val, lane_mask);
    int other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, lane_mask);
    
    int lane_id = threadIdx.x & 31;
    bool is_lower_lane = (lane_id & lane_mask) == 0;
    
    // For descending: lower lane keeps max, higher lane keeps min
    bool swap = dir ? (is_lower_lane ? (val < other_val) : (val > other_val))
                    : (is_lower_lane ? (val > other_val) : (val < other_val));
    
    if (swap) {
        val = other_val;
        idx = other_idx;
    }
}

// Warp-level bitonic sort for exactly 32 elements (one per lane)
// Sorts in descending order (largest first)
__device__ __forceinline__ void warp_bitonic_sort_32(float& val, int& idx) {
    // Bitonic sort network for 32 elements
    // 5 stages: k = 2, 4, 8, 16, 32
    
    // Stage 1: k=2 (pairs)
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 2: k=4
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 3: k=8
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 4: k=16
    warp_compare_swap(val, idx, 8, true);
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
    
    // Stage 5: k=32
    warp_compare_swap(val, idx, 16, true);
    warp_compare_swap(val, idx, 8, true);
    warp_compare_swap(val, idx, 4, true);
    warp_compare_swap(val, idx, 2, true);
    warp_compare_swap(val, idx, 1, true);
}

// Warp-level bitonic sort for n ≤ 32 elements
// Elements beyond n are set to -INFINITY to sort to the end
__device__ __forceinline__ void warp_bitonic_sort_n(
    float& val, int& idx, int n
) {
    int lane_id = threadIdx.x & 31;
    
    // Invalid lanes get -INFINITY (will sort to end in descending order)
    if (lane_id >= n) {
        val = -INFINITY;
        idx = -1;
    }
    
    // Full 32-element sort (invalid elements naturally go to end)
    warp_bitonic_sort_32(val, idx);
}

// Sort top-k candidates using warp shuffles (for k ≤ 32)
// Loads from shared memory, sorts in registers, writes back
__device__ __forceinline__ void warp_sort_topk(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    int n  // Must be ≤ 32
) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    
    // Only first warp does the sort
    if (warp_id == 0) {
        // Load into registers
        float my_val = (lane_id < n) ? vals[lane_id] : -INFINITY;
        int my_idx = (lane_id < n) ? idxs[lane_id] : -1;
        
        // Sort using warp shuffles
        warp_bitonic_sort_32(my_val, my_idx);
        
        // Write back sorted results
        if (lane_id < n) {
            vals[lane_id] = my_val;
            idxs[lane_id] = my_idx;
        }
    }
}

// Merge two sorted sequences of length k using warp shuffles
// Both sequences must be in registers across lanes 0..k-1 and k..2k-1
__device__ __forceinline__ void warp_bitonic_merge(
    float& val, int& idx, int k
) {
    // Bitonic merge: compare lanes i and i+k, then sort each half
    int lane_id = threadIdx.x & 31;
    
    if (k <= 16) {
        // Exchange between halves
        warp_compare_swap(val, idx, k, true);
        
        // Sort within each half
        for (int j = k / 2; j >= 1; j /= 2) {
            warp_compare_swap(val, idx, j, true);
        }
    }
}

// Parallel top-k selection using warp-level operations
// Each warp maintains its own sorted top-k candidates, then merges
template <int THREADS>
__device__ int warp_parallel_topk(
    float* __restrict__ smem_vals,      // [MAX_K] shared memory for values
    int* __restrict__ smem_idxs,        // [MAX_K] shared memory for indices
    int current_count,                   // Current number of candidates
    int target_k                         // Target k (must be ≤ 32)
) {
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid / 32;
    const int num_warps = THREADS / 32;
    
    int actual_k = min(current_count, target_k);
    
    if (actual_k <= 32 && warp_id == 0) {
        // Simple case: just sort with first warp
        float my_val = (lane_id < actual_k) ? smem_vals[lane_id] : -INFINITY;
        int my_idx = (lane_id < actual_k) ? smem_idxs[lane_id] : -1;
        
        warp_bitonic_sort_32(my_val, my_idx);
        
        if (lane_id < actual_k) {
            smem_vals[lane_id] = my_val;
            smem_idxs[lane_id] = my_idx;
        }
    }
    else if (actual_k > 32) {
        // Multi-warp case: each warp sorts a chunk, then merge
        // For now, fall through to regular bitonic sort
        return -1;  // Signal to use regular bitonic sort
    }
    
    return actual_k;
}

// ============================================================================
// Hybrid Sort: Warp shuffle for small k, shared memory for large k
// ============================================================================

// Helper: round up to next power of 2
__device__ __forceinline__ int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Bitonic sort for small arrays (used after radix select narrows candidates)
// Handles non-power-of-2 sizes by padding with -INFINITY sentinels
__device__ __forceinline__ void bitonic_sort_descending(
    float* __restrict__ vals,
    int* __restrict__ idxs,
    int n
) {
    // Use warp shuffle sort for n ≤ 32 (much faster)
    if (n <= 32) {
        warp_sort_topk(vals, idxs, n);
        __syncthreads();
        return;
    }
    
    // For bitonic sort, we need to work with power-of-2 sizes
    // Pad with -INFINITY sentinels that will sort to the end (descending)
    int padded_n = next_power_of_2(n);
    
    // Pad with sentinels (only threads with indices in [n, padded_n) do this)
    for (int i = n + threadIdx.x; i < padded_n; i += blockDim.x) {
        vals[i] = -INFINITY;
        idxs[i] = -1;  // Invalid index
    }
    __syncthreads();
    
    // Standard bitonic sort on padded array
    for (int k = 2; k <= padded_n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = threadIdx.x; i < padded_n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    // Descending order: swap if vals[i] < vals[ixj] in first half of bitonic sequence
                    bool ascending = ((i & k) == 0);
                    bool swap = ascending ? (vals[i] < vals[ixj]) : (vals[i] > vals[ixj]);
                    if (swap) {
                        float tmp_v = vals[i]; vals[i] = vals[ixj]; vals[ixj] = tmp_v;
                        int tmp_i = idxs[i]; idxs[i] = idxs[ixj]; idxs[ixj] = tmp_i;
                    }
                }
            }
            __syncthreads();
        }
    }
    // After sorting, vals[0..n-1] contains the sorted data, vals[n..padded_n-1] = -INFINITY
}

// ============================================================================
// Radix Select for O(n) Top-K Selection
// ============================================================================

// Convert float to sortable uint32 (handles negative numbers correctly)
__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t bits = __float_as_uint(f);
    // Flip sign bit for positive, flip all bits for negative
    uint32_t mask = -int32_t(bits >> 31) | 0x80000000;
    return bits ^ mask;
}

// Convert sortable uint32 back to float
__device__ __forceinline__ float sortable_to_float(uint32_t u) {
    uint32_t mask = ((u >> 31) - 1) | 0x80000000;
    return __uint_as_float(u ^ mask);
}
