// Batched GPU-native multinomial sampling kernel
// Based on PyTorch's MultinomialKernel.cu implementation
// Processes multiple samples in parallel

#include "cuda_utils.cuh"
#include <stdint.h>

#define MASK_VAL -1e38f

// Binary search for multinomial sampling (from PyTorch)
template<typename T>
__device__ int binarySearchForMultinomial(
    const T* cumdist,
    int size,
    T val
) {
    int start = 0;
    int end = size;
    
    // Handle edge case
    if (cumdist[size - 1] <= static_cast<T>(0)) {
        return 0;
    }
    
    while (end - start > 0) {
        int mid = start + (end - start) / 2;
        T midVal = cumdist[mid];
        
        if (midVal < val) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    
    if (start == size) {
        start = size - 1;
    }
    
    return start;
}

// Batched multinomial sampling kernel
// Each block processes one sample independently
template<typename T>
__device__ void multinomial_batch_kernel_impl(
    const T* logits,              // [batch_size, vocab_size]
    uint32_t* output,             // [batch_size]
    const float* random_samples,  // [batch_size] - pre-generated uniform[0,1) samples
    const size_t batch_size,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p
) {
    // Each block handles one sample
    const int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) return;
    
    // Thread ID within block
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Use shared memory for parallel operations
    extern __shared__ float smem[];
    float* probs = smem;              // vocab_size floats
    float* cumsum_buf = smem + vocab_size;  // vocab_size floats
    
    const T* sample_logits = logits + sample_idx * vocab_size;
    const float rand_val = random_samples[sample_idx];
    
    // Step 1: Apply temperature and compute softmax in parallel
    // Find max for numerical stability
    float thread_max = -1e38f;  // Large negative number instead of -INFINITY
    for (size_t i = tid; i < vocab_size; i += num_threads) {
        float val = static_cast<float>(sample_logits[i]) / temperature;
        if (val > thread_max) {
            thread_max = val;
        }
    }
    
    // Store thread max in shared memory for reduction
    cumsum_buf[tid] = thread_max;
    __syncthreads();
    
    // Block-wide reduction to find global max
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_threads) {
            cumsum_buf[tid] = fmaxf(cumsum_buf[tid], cumsum_buf[tid + s]);
        }
        __syncthreads();
    }
    
    const float max_logit = cumsum_buf[0];
    __syncthreads();
    
    // Compute exp(logit/temp - max) and local sum
    float thread_sum = 0.0f;
    for (size_t i = tid; i < vocab_size; i += num_threads) {
        float val = expf(static_cast<float>(sample_logits[i]) / temperature - max_logit);
        probs[i] = val;
        thread_sum += val;
    }
    
    // Store thread sum for reduction
    cumsum_buf[tid] = thread_sum;
    __syncthreads();
    
    // Block-wide reduction to compute total sum
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_threads) {
            cumsum_buf[tid] = cumsum_buf[tid] + cumsum_buf[tid + s];
        }
        __syncthreads();
    }
    
    const float sum_exp = cumsum_buf[0];
    __syncthreads();
    
    // Normalize to get probabilities
    for (size_t i = tid; i < vocab_size; i += num_threads) {
        probs[i] /= sum_exp;
    }
    __syncthreads();
    
    // Step 2: Compute cumulative distribution (prefix sum)
    // Simple sequential approach (can be optimized with parallel scan)
    if (tid == 0) {
        float cumsum = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            cumsum += probs[i];
            cumsum_buf[i] = cumsum;
        }
    }
    __syncthreads();
    
    // Step 3: Binary search to find sample
    if (tid == 0) {
        int sampled_idx = binarySearchForMultinomial(cumsum_buf, vocab_size, rand_val);
        output[sample_idx] = static_cast<uint32_t>(sampled_idx);
    }
}

// Kernel launchers for different data types
#define MULTINOMIAL_BATCH_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME* logits, \
    uint32_t* output, \
    const float* random_samples, \
    const size_t batch_size, \
    const size_t vocab_size, \
    const float temperature, \
    const uint32_t top_k, \
    const float top_p \
) { \
    multinomial_batch_kernel_impl<TYPENAME>( \
        logits, output, random_samples, \
        batch_size, vocab_size, temperature, top_k, top_p \
    ); \
}

MULTINOMIAL_BATCH_OP(float, multinomial_batch_f32)
MULTINOMIAL_BATCH_OP(double, multinomial_batch_f64)
MULTINOMIAL_BATCH_OP(__half, multinomial_batch_f16)

#if __CUDA_ARCH__ >= 800
MULTINOMIAL_BATCH_OP(__nv_bfloat16, multinomial_batch_bf16)
#endif
