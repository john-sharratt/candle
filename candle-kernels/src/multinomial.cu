#include "cuda_utils.cuh"
#include <curand_kernel.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

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
        // Use multiplication instead of division
        local_sum += expf((val - max_logit) * inv_temp);
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

// F32 optimized kernel
extern "C" __global__ void optimized_multinomial_f32(
    const float* logits,
    uint32_t* output,
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
        
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = logits[i];
            float prob = expf((score - max_logit) / temperature) / sum_exp;
            cumulative += prob;
            if (rand_val <= cumulative) {
                *output = static_cast<uint32_t>(i);
                return;
            }
        }
        *output = static_cast<uint32_t>(vocab_size - 1);
    }
}

// F64 optimized kernel
extern "C" __global__ void optimized_multinomial_f64(
    const double* logits,
    uint32_t* output,
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
        
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = static_cast<float>(logits[i]);
            float prob = expf((score - max_logit) / temperature) / sum_exp;
            cumulative += prob;
            if (rand_val <= cumulative) {
                *output = static_cast<uint32_t>(i);
                return;
            }
        }
        *output = static_cast<uint32_t>(vocab_size - 1);
    }
}

// F16 optimized kernel
extern "C" __global__ void optimized_multinomial_f16(
    const __half* logits,
    uint32_t* output,
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
        
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = __half2float(logits[i]);
            float prob = expf((score - max_logit) / temperature) / sum_exp;
            cumulative += prob;
            if (rand_val <= cumulative) {
                *output = static_cast<uint32_t>(i);
                return;
            }
        }
        *output = static_cast<uint32_t>(vocab_size - 1);
    }
}

// BF16 optimized kernel
#if __CUDA_ARCH__ >= 800
extern "C" __global__ void optimized_multinomial_bf16(
    const __nv_bfloat16* logits,
    uint32_t* output,
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
        
        float cumulative = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            float score = __bfloat162float(logits[i]);
            float prob = expf((score - max_logit) / temperature) / sum_exp;
            cumulative += prob;
            if (rand_val <= cumulative) {
                *output = static_cast<uint32_t>(i);
                return;
            }
        }
        *output = static_cast<uint32_t>(vocab_size - 1);
    }
}
#else
extern "C" __global__ void optimized_multinomial_bf16(
    const __nv_bfloat16* logits,
    uint32_t* output,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
) {
    // bf16 not supported on pre-Ampere
}
#endif
