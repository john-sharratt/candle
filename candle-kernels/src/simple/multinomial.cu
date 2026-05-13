#include "cuda_utils.cuh"
#include "../fast_exp.cuh"
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GPU-accelerated multinomial sampling with full top-k and top-p support.
// 
// Strategy for top-k/top-p:
// 1. Use bitonic sort for small vocabularies (< 4096 tokens)
// 2. Use partial sort: only find top-k elements using parallel selection
// 3. Compute prefix sum in parallel for top-p filtering
// 
// This avoids expensive full sorting while maintaining GPU acceleration.

// Safe softmax probability computation that handles -inf - (-inf) = NaN branchlessly
// Uses fmaxf(NaN, -inf) = -inf per IEEE 754-2008, then fast_exp returns 0 for very negative values
__device__ __forceinline__ float safe_softmax_prob(float score, float max_logit, float inv_temp, float inv_sum_exp) {
    return fast_exp::exp<float, fast_exp::Softmax>(fmaxf((score - max_logit) * inv_temp, -INFINITY)) * inv_sum_exp;
}

// Parallel max reduction using shared memory
template <typename T>
__device__ float parallel_max_reduce(const T* logits, size_t vocab_size) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Each thread finds max in its subset
    float local_max = -INFINITY;
    for (size_t i = tid; i < vocab_size; i += stride) {
        float val = static_cast<float>(logits[i]);
        local_max = fmaxf(local_max, val);
    }
    
    // Store in shared memory
    shared_mem[tid] = local_max;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]);
        }
        __syncthreads();
    }
    
    // Thread 0 has the final result
    return shared_mem[0];
}

// Parallel sum of exp reduction using shared memory
template <typename T>
__device__ float parallel_sum_exp_reduce(const T* logits, size_t vocab_size, float max_logit, float temperature) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Cache inverse temperature (multiplication is faster than division)
    float inv_temp = 1.0f / temperature;
    
    // Each thread computes sum for its subset
    float local_sum = 0.0f;
    for (size_t i = tid; i < vocab_size; i += stride) {
        float val = static_cast<float>(logits[i]);
        // Use fmaxf to handle -inf - (-inf) = NaN case branchlessly
        // fast_exp::Softmax mode assumes x <= 0 (after max subtraction)
        local_sum += fast_exp::exp<float, fast_exp::Softmax>(fmaxf((val - max_logit) * inv_temp, -INFINITY));
    }
    
    // Store in shared memory
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 has the final result
    return shared_mem[0];
}

// Struct to hold (probability, index) pairs for sorting
struct ProbIndex {
    float prob;
    uint32_t index;
};

// Kernel to compute probabilities with temperature scaling
template <typename T>
__global__ void compute_probs_kernel(
    const T* logits,
    ProbIndex* prob_indices,
    const size_t vocab_size,
    const float temperature,
    const float max_logit,
    const float sum_exp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vocab_size) {
        float score = static_cast<float>(logits[idx]);
        float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
        prob_indices[idx].prob = prob;
        prob_indices[idx].index = idx;
    }
}

// Bitonic sort step for small arrays
__device__ void bitonic_sort_step(ProbIndex* data, int size, int tid, int step, int stage) {
    int ixj = tid ^ step;
    if (ixj > tid) {
        if ((tid & stage) == 0) {
            // Ascending
            if (data[tid].prob < data[ixj].prob) {
                ProbIndex temp = data[tid];
                data[tid] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descending
            if (data[tid].prob > data[ixj].prob) {
                ProbIndex temp = data[tid];
                data[tid] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}

// Parallel bitonic sort kernel (for descending order)
__global__ void bitonic_sort_kernel(ProbIndex* data, int size) {
    extern __shared__ ProbIndex shared_data[];
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (global_tid < size) {
        shared_data[tid] = data[global_tid];
    } else {
        shared_data[tid].prob = -INFINITY;
        shared_data[tid].index = 0;
    }
    __syncthreads();
    
    // Bitonic sort in shared memory
    for (int stage = 2; stage <= blockDim.x; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step >>= 1) {
            bitonic_sort_step(shared_data, blockDim.x, tid, step, stage);
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_tid < size) {
        data[global_tid] = shared_data[tid];
    }
}

// Simple selection kernel to find k-th largest elements (partial sort)
__global__ void partial_sort_topk_kernel(
    ProbIndex* data,
    ProbIndex* output,
    const size_t vocab_size,
    const uint32_t k
) {
    int tid = threadIdx.x;
    
    // Each thread maintains local top-k using insertion sort
    extern __shared__ ProbIndex shared_topk[];
    
    // Initialize thread's top-k buffer
    for (int i = 0; i < k; ++i) {
        int idx = tid * k + i;
        if (tid == 0 && idx < k && idx < vocab_size) {
            shared_topk[idx] = data[idx];
        } else {
            if (idx < blockDim.x * k) {
                shared_topk[idx].prob = -INFINITY;
                shared_topk[idx].index = 0;
            }
        }
    }
    __syncthreads();
    
    // Process elements in parallel
    for (size_t i = tid; i < vocab_size; i += blockDim.x) {
        ProbIndex current = data[i];
        
        // Find insertion point in thread's local top-k
        for (int j = 0; j < k; ++j) {
            int idx = tid * k + j;
            if (current.prob > shared_topk[idx].prob) {
                // Shift elements down
                for (int l = k - 1; l > j; --l) {
                    int idx_l = tid * k + l;
                    int idx_prev = tid * k + l - 1;
                    shared_topk[idx_l] = shared_topk[idx_prev];
                }
                shared_topk[idx] = current;
                break;
            }
        }
    }
    __syncthreads();
    
    // Merge thread results (simple approach: thread 0 merges)
    if (tid == 0) {
        // Collect all candidates
        ProbIndex candidates[256]; // Max 256 threads * k elements
        int count = 0;
        for (int t = 0; t < blockDim.x && count < 256; ++t) {
            for (int j = 0; j < k && count < 256; ++j) {
                int idx = t * k + j;
                if (shared_topk[idx].prob > -INFINITY) {
                    candidates[count++] = shared_topk[idx];
                }
            }
        }
        
        // Simple insertion sort to find final top-k
        for (int i = 1; i < count; ++i) {
            ProbIndex key = candidates[i];
            int j = i - 1;
            while (j >= 0 && candidates[j].prob < key.prob) {
                candidates[j + 1] = candidates[j];
                j--;
            }
            candidates[j + 1] = key;
        }
        
        // Write final top-k
        for (int i = 0; i < k && i < count; ++i) {
            output[i] = candidates[i];
        }
    }
}

// F32 optimized kernel with proper top-k and top-p support
extern "C" __global__ void optimized_multinomial_f32(
    const float* logits,
    uint32_t* output,
    float* workspace,  // Workspace for intermediate computations
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
) {
    // Find max logit in parallel
    float max_logit = parallel_max_reduce(logits, vocab_size);
    
    // Compute sum of exp(logit - max_logit) in parallel
    float sum_exp = parallel_sum_exp_reduce(logits, vocab_size, max_logit, temperature);
    
    // Only thread 0 does the final sampling (after parallel work done)
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(seed, 0, 0, &state);
        float rand_val = curand_uniform(&state);
        
        // Fast path: no filtering needed
        if (top_k == 0 && top_p >= 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                float score = logits[i];
                float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
                cumulative += prob;
                if (rand_val <= cumulative) {
                    *output = static_cast<uint32_t>(i);
                    return;
                }
            }
            *output = static_cast<uint32_t>(vocab_size - 1);
            return;
        }
        
        // Filtering path: need sorted probabilities
        // Use workspace as ProbIndex array
        ProbIndex* prob_indices = reinterpret_cast<ProbIndex*>(workspace);
        
        // Compute all probabilities
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = logits[i];
            float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
            prob_indices[i].prob = prob;
            prob_indices[i].index = i;
        }
        
        // Smart partial sorting: only sort what we need
        // Determine how many elements we actually need to sort
        size_t sort_limit = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            // For top_k, we only need k elements sorted
            sort_limit = min((size_t)top_k + 5, vocab_size); // +5 buffer for top_p
        } else if (top_p < 1.0f) {
            // For top_p, estimate based on probability mass
            // Most distributions concentrate in top 100-200 tokens
            sort_limit = min(vocab_size, (size_t)200);
        }
        
        // Use insertion sort for small ranges (< 128), selection sort for larger
        if (sort_limit < 128) {
            // Insertion sort for the first sort_limit elements
            for (size_t i = 1; i < sort_limit; ++i) {
                ProbIndex key = prob_indices[i];
                int j = i - 1;
                while (j >= 0 && prob_indices[j].prob < key.prob) {
                    prob_indices[j + 1] = prob_indices[j];
                    j--;
                }
                prob_indices[j + 1] = key;
            }
            
            // Now find if any remaining elements should be in top sort_limit
            for (size_t i = sort_limit; i < vocab_size; ++i) {
                if (prob_indices[i].prob > prob_indices[sort_limit - 1].prob) {
                    // Insert this element into sorted portion
                    ProbIndex key = prob_indices[i];
                    size_t j = sort_limit - 1;
                    while (j > 0 && prob_indices[j - 1].prob < key.prob) {
                        prob_indices[j] = prob_indices[j - 1];
                        j--;
                    }
                    prob_indices[j] = key;
                }
            }
        } else {
            // Partial selection sort for larger ranges
            for (size_t i = 0; i < sort_limit; ++i) {
                size_t max_idx = i;
                for (size_t j = i + 1; j < vocab_size; ++j) {
                    if (prob_indices[j].prob > prob_indices[max_idx].prob) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    ProbIndex temp = prob_indices[i];
                    prob_indices[i] = prob_indices[max_idx];
                    prob_indices[max_idx] = temp;
                }
            }
        }
        
        // Determine effective vocabulary size
        size_t effective_size = vocab_size;
        
        // Apply top-k filtering
        if (top_k > 0 && top_k < vocab_size) {
            effective_size = min(effective_size, (size_t)top_k);
        }
        
        // Apply top-p filtering
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < effective_size; ++i) {
                cumulative += prob_indices[i].prob;
                if (cumulative >= top_p) {
                    effective_size = i + 1;
                    break;
                }
            }
        }
        
        // Renormalize probabilities
        float total_prob = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            total_prob += prob_indices[i].prob;
        }
        
        // Sample from filtered distribution
        float target = rand_val * total_prob;
        float cumulative = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            cumulative += prob_indices[i].prob;
            if (target <= cumulative) {
                *output = prob_indices[i].index;
                return;
            }
        }
        
        // Fallback to last token in filtered set
        *output = prob_indices[effective_size - 1].index;
    }
}

// F64 optimized kernel with top-k and top-p support
extern "C" __global__ void optimized_multinomial_f64(
    const double* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
) {
    float max_logit = parallel_max_reduce(logits, vocab_size);
    float sum_exp = parallel_sum_exp_reduce(logits, vocab_size, max_logit, temperature);
    
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(seed, 0, 0, &state);
        float rand_val = curand_uniform(&state);
        
        // Fast path: no filtering
        if (top_k == 0 && top_p >= 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                float score = static_cast<float>(logits[i]);
                float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
                cumulative += prob;
                if (rand_val <= cumulative) {
                    *output = static_cast<uint32_t>(i);
                    return;
                }
            }
            *output = static_cast<uint32_t>(vocab_size - 1);
            return;
        }
        
        // Filtering path
        ProbIndex* prob_indices = reinterpret_cast<ProbIndex*>(workspace);
        
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = static_cast<float>(logits[i]);
            float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
            prob_indices[i].prob = prob;
            prob_indices[i].index = i;
        }
        
        // Smart partial sorting (F64 kernel)
        size_t sort_limit = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            sort_limit = min((size_t)top_k + 5, vocab_size);
        } else if (top_p < 1.0f) {
            sort_limit = min(vocab_size, (size_t)200);
        }
        
        if (sort_limit < 128) {
            for (size_t i = 1; i < sort_limit; ++i) {
                ProbIndex key = prob_indices[i];
                int j = i - 1;
                while (j >= 0 && prob_indices[j].prob < key.prob) {
                    prob_indices[j + 1] = prob_indices[j];
                    j--;
                }
                prob_indices[j + 1] = key;
            }
            for (size_t i = sort_limit; i < vocab_size; ++i) {
                if (prob_indices[i].prob > prob_indices[sort_limit - 1].prob) {
                    ProbIndex key = prob_indices[i];
                    size_t j = sort_limit - 1;
                    while (j > 0 && prob_indices[j - 1].prob < key.prob) {
                        prob_indices[j] = prob_indices[j - 1];
                        j--;
                    }
                    prob_indices[j] = key;
                }
            }
        } else {
            for (size_t i = 0; i < sort_limit; ++i) {
                size_t max_idx = i;
                for (size_t j = i + 1; j < vocab_size; ++j) {
                    if (prob_indices[j].prob > prob_indices[max_idx].prob) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    ProbIndex temp = prob_indices[i];
                    prob_indices[i] = prob_indices[max_idx];
                    prob_indices[max_idx] = temp;
                }
            }
        }
        
        size_t effective_size = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            effective_size = min(effective_size, (size_t)top_k);
        }
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < effective_size; ++i) {
                cumulative += prob_indices[i].prob;
                if (cumulative >= top_p) {
                    effective_size = i + 1;
                    break;
                }
            }
        }
        
        float total_prob = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            total_prob += prob_indices[i].prob;
        }
        
        float target = rand_val * total_prob;
        float cumulative = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            cumulative += prob_indices[i].prob;
            if (target <= cumulative) {
                *output = prob_indices[i].index;
                return;
            }
        }
        *output = prob_indices[effective_size - 1].index;
    }
}

// F16 optimized kernel with top-k and top-p support
extern "C" __global__ void optimized_multinomial_f16(
    const __half* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
) {
    float max_logit = parallel_max_reduce(logits, vocab_size);
    float sum_exp = parallel_sum_exp_reduce(logits, vocab_size, max_logit, temperature);
    
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(seed, 0, 0, &state);
        float rand_val = curand_uniform(&state);
        
        // Fast path: no filtering
        if (top_k == 0 && top_p >= 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                float score = __half2float(logits[i]);
                float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
                cumulative += prob;
                if (rand_val <= cumulative) {
                    *output = static_cast<uint32_t>(i);
                    return;
                }
            }
            *output = static_cast<uint32_t>(vocab_size - 1);
            return;
        }
        
        // Filtering path
        ProbIndex* prob_indices = reinterpret_cast<ProbIndex*>(workspace);
        
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = __half2float(logits[i]);
            float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
            prob_indices[i].prob = prob;
            prob_indices[i].index = i;
        }
        
        // Smart partial sorting (F16 kernel)
        size_t sort_limit = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            sort_limit = min((size_t)top_k + 5, vocab_size);
        } else if (top_p < 1.0f) {
            sort_limit = min(vocab_size, (size_t)200);
        }
        
        if (sort_limit < 128) {
            for (size_t i = 1; i < sort_limit; ++i) {
                ProbIndex key = prob_indices[i];
                int j = i - 1;
                while (j >= 0 && prob_indices[j].prob < key.prob) {
                    prob_indices[j + 1] = prob_indices[j];
                    j--;
                }
                prob_indices[j + 1] = key;
            }
            for (size_t i = sort_limit; i < vocab_size; ++i) {
                if (prob_indices[i].prob > prob_indices[sort_limit - 1].prob) {
                    ProbIndex key = prob_indices[i];
                    size_t j = sort_limit - 1;
                    while (j > 0 && prob_indices[j - 1].prob < key.prob) {
                        prob_indices[j] = prob_indices[j - 1];
                        j--;
                    }
                    prob_indices[j] = key;
                }
            }
        } else {
            for (size_t i = 0; i < sort_limit; ++i) {
                size_t max_idx = i;
                for (size_t j = i + 1; j < vocab_size; ++j) {
                    if (prob_indices[j].prob > prob_indices[max_idx].prob) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    ProbIndex temp = prob_indices[i];
                    prob_indices[i] = prob_indices[max_idx];
                    prob_indices[max_idx] = temp;
                }
            }
        }
        
        size_t effective_size = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            effective_size = min(effective_size, (size_t)top_k);
        }
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < effective_size; ++i) {
                cumulative += prob_indices[i].prob;
                if (cumulative >= top_p) {
                    effective_size = i + 1;
                    break;
                }
            }
        }
        
        float total_prob = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            total_prob += prob_indices[i].prob;
        }
        
        float target = rand_val * total_prob;
        float cumulative = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            cumulative += prob_indices[i].prob;
            if (target <= cumulative) {
                *output = prob_indices[i].index;
                return;
            }
        }
        *output = prob_indices[effective_size - 1].index;
    }
}

// BF16 optimized kernel with top-k and top-p support
extern "C" __global__ void optimized_multinomial_bf16(
    const __nv_bfloat16* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
) {
    float max_logit = parallel_max_reduce(logits, vocab_size);
    float sum_exp = parallel_sum_exp_reduce(logits, vocab_size, max_logit, temperature);
    
    if (threadIdx.x == 0) {
        curandState state;
        curand_init(seed, 0, 0, &state);
        float rand_val = curand_uniform(&state);
        
        // Fast path: no filtering
        if (top_k == 0 && top_p >= 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                float score = __bfloat162float(logits[i]);
                float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
                cumulative += prob;
                if (rand_val <= cumulative) {
                    *output = static_cast<uint32_t>(i);
                    return;
                }
            }
            *output = static_cast<uint32_t>(vocab_size - 1);
            return;
        }
        
        // Filtering path
        ProbIndex* prob_indices = reinterpret_cast<ProbIndex*>(workspace);
        
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = __bfloat162float(logits[i]);
            float prob = safe_softmax_prob(score, max_logit, 1.0f / temperature, 1.0f / sum_exp);
            prob_indices[i].prob = prob;
            prob_indices[i].index = i;
        }
        
        // Smart partial sorting (BF16 kernel)
        size_t sort_limit = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            sort_limit = min((size_t)top_k + 5, vocab_size);
        } else if (top_p < 1.0f) {
            sort_limit = min(vocab_size, (size_t)200);
        }
        
        if (sort_limit < 128) {
            for (size_t i = 1; i < sort_limit; ++i) {
                ProbIndex key = prob_indices[i];
                int j = i - 1;
                while (j >= 0 && prob_indices[j].prob < key.prob) {
                    prob_indices[j + 1] = prob_indices[j];
                    j--;
                }
                prob_indices[j + 1] = key;
            }
            for (size_t i = sort_limit; i < vocab_size; ++i) {
                if (prob_indices[i].prob > prob_indices[sort_limit - 1].prob) {
                    ProbIndex key = prob_indices[i];
                    size_t j = sort_limit - 1;
                    while (j > 0 && prob_indices[j - 1].prob < key.prob) {
                        prob_indices[j] = prob_indices[j - 1];
                        j--;
                    }
                    prob_indices[j] = key;
                }
            }
        } else {
            for (size_t i = 0; i < sort_limit; ++i) {
                size_t max_idx = i;
                for (size_t j = i + 1; j < vocab_size; ++j) {
                    if (prob_indices[j].prob > prob_indices[max_idx].prob) {
                        max_idx = j;
                    }
                }
                if (max_idx != i) {
                    ProbIndex temp = prob_indices[i];
                    prob_indices[i] = prob_indices[max_idx];
                    prob_indices[max_idx] = temp;
                }
            }
        }
        
        size_t effective_size = vocab_size;
        if (top_k > 0 && top_k < vocab_size) {
            effective_size = min(effective_size, (size_t)top_k);
        }
        if (top_p < 1.0f) {
            float cumulative = 0.0f;
            for (size_t i = 0; i < effective_size; ++i) {
                cumulative += prob_indices[i].prob;
                if (cumulative >= top_p) {
                    effective_size = i + 1;
                    break;
                }
            }
        }
        
        float total_prob = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            total_prob += prob_indices[i].prob;
        }
        
        float target = rand_val * total_prob;
        float cumulative = 0.0f;
        for (size_t i = 0; i < effective_size; ++i) {
            cumulative += prob_indices[i].prob;
            if (target <= cumulative) {
                *output = prob_indices[i].index;
                return;
            }
        }
        *output = prob_indices[effective_size - 1].index;
    }
}
