// Simple multinomial sampling kernel for large vocabularies
// Uses global memory instead of shared memory to support vocab > 4K

#include "cuda_utils.cuh"
#include <stdint.h>

// Simple LCG random number generator
__device__ float lcg_random(uint64_t* seed) {
    *seed = (*seed * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    return (*seed % 1000000) / 1000000.0f;
}

// Simple kernel that does softmax + sampling without top-k/top-p for now
// Works with any vocabulary size (uses global memory)
template<typename T>
__device__ void simple_multinomial_kernel(
    const T* logits,
    uint32_t* output,
    const size_t vocab_size,
    const float temperature,
    const uint64_t seed
) {
    uint64_t rng_state = seed;
    
    // Step 1: Find max for numerical stability (simple sequential)
    float max_logit = -1e38f;
    for (size_t i = 0; i < vocab_size; i++) {
        float val = static_cast<float>(logits[i]) / temperature;
        if (val > max_logit) {
            max_logit = val;
        }
    }
    
    // Step 2: Compute sum of exp(logit - max)
    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        float val = static_cast<float>(logits[i]) / temperature;
        sum_exp += expf(val - max_logit);
    }
    
    // Step 3: Generate random number and sample
    float rand_val = lcg_random(&rng_state);
    float target = rand_val * sum_exp;
    
    float cumsum = 0.0f;
    uint32_t sampled_idx = vocab_size - 1;
    
    for (size_t i = 0; i < vocab_size; i++) {
        float val = static_cast<float>(logits[i]) / temperature;
        float prob = expf(val - max_logit);
        cumsum += prob;
        
        if (cumsum >= target) {
            sampled_idx = i;
            break;
        }
    }
    
    output[0] = sampled_idx;
}

// Kernel launchers
#define SIMPLE_MULTINOMIAL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME* logits, \
    uint32_t* output, \
    const size_t vocab_size, \
    const float temperature, \
    const uint64_t seed \
) { \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        simple_multinomial_kernel<TYPENAME>( \
            logits, output, vocab_size, temperature, seed \
        ); \
    } \
}

SIMPLE_MULTINOMIAL_OP(float, simple_multinomial_f32)
SIMPLE_MULTINOMIAL_OP(double, simple_multinomial_f64)
SIMPLE_MULTINOMIAL_OP(__half, simple_multinomial_f16)

#if __CUDA_ARCH__ >= 800
SIMPLE_MULTINOMIAL_OP(__nv_bfloat16, simple_multinomial_bf16)
#endif
