// GPU-native multinomial sampling kernel
// Eliminates all GPU→CPU transfers during sampling

#include "cuda_utils.cuh"
#include <stdint.h>

// Define constants for masking invalid values
#define MASK_VAL -1e38f

// Simple LCG random number generator (GPU-friendly)
__device__ float lcg_random(uint64_t* seed) {
    *seed = (*seed * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    return (*seed % 1000000) / 1000000.0f;
}

// GPU-native multinomial sampling kernel
// Device function called by wrapper kernels
template<typename T>
__device__ void multinomial_sampling_kernel(
    const T* logits,           // Input logits [vocab_size]
    uint32_t* output,          // Output token ID [1]
    const size_t vocab_size,   // Vocabulary size
    const float temperature,   // Temperature for scaling
    const uint32_t top_k,      // Top-k filtering (0 = disabled)
    const float top_p,         // Top-p/nucleus sampling (0.0 = disabled)  
    const uint64_t seed        // Random seed
) {
    // We only need one thread for this operation - already checked in wrapper
    // if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Only first thread does the work (for now - can be parallelized later)
    // This is acceptable for single-sample processing
    
    // Allocate shared memory for processing
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;                    // vocab_size floats
    float* probs = &shared_mem[vocab_size];        // vocab_size floats
    // Align indices to uint32 boundary (shared_mem is float-aligned, which is fine for uint32)
    uint32_t* indices = (uint32_t*)&shared_mem[vocab_size * 2]; // vocab_size uints
    
    // Initialize random seed
    uint64_t rng_state = seed;
    
    // Step 1: Apply temperature scaling and convert to float
    for (size_t i = 0; i < vocab_size; i++) {
        scores[i] = static_cast<float>(logits[i]) / temperature;
        indices[i] = i;
    }
    
    // Step 2: Apply top-k filtering if enabled
    size_t active_size = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        // Partial sort to get top-k elements
        // Simple selection algorithm (can be optimized with parallel sorting)
        for (uint32_t k = 0; k < top_k; k++) {
            // Find k-th largest element
            float max_val = MASK_VAL;
            size_t max_idx = k;
            for (size_t i = k; i < vocab_size; i++) {
                if (scores[i] > max_val) {
                    max_val = scores[i];
                    max_idx = i;
                }
            }
            // Swap with position k
            float tmp_score = scores[k];
            scores[k] = scores[max_idx];
            scores[max_idx] = tmp_score;
            
            uint32_t tmp_idx = indices[k];
            indices[k] = indices[max_idx];
            indices[max_idx] = tmp_idx;
        }
        
        // Mask out elements beyond top-k
        for (size_t i = top_k; i < vocab_size; i++) {
            scores[i] = MASK_VAL;
        }
        active_size = top_k;
    }
    
    // Step 3: Convert to probabilities (softmax)
    float max_score = MASK_VAL;
    for (size_t i = 0; i < vocab_size; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        if (scores[i] != MASK_VAL) {
            probs[i] = expf(scores[i] - max_score);
            sum += probs[i];
        } else {
            probs[i] = 0.0f;
        }
    }
    
    // Normalize
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Step 4: Apply top-p (nucleus) sampling if enabled
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort by probability (descending)
        for (size_t i = 0; i < active_size - 1; i++) {
            for (size_t j = i + 1; j < active_size; j++) {
                if (probs[j] > probs[i]) {
                    float tmp_prob = probs[i];
                    probs[i] = probs[j];
                    probs[j] = tmp_prob;
                    
                    uint32_t tmp_idx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tmp_idx;
                }
            }
        }
        
        // Find cutoff point
        float cumulative = 0.0f;
        size_t cutoff = active_size;
        for (size_t i = 0; i < active_size; i++) {
            cumulative += probs[i];
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (size_t i = cutoff; i < vocab_size; i++) {
            probs[i] = 0.0f;
        }
        
        // Renormalize
        sum = 0.0f;
        for (size_t i = 0; i < cutoff; i++) {
            sum += probs[i];
        }
        for (size_t i = 0; i < cutoff; i++) {
            probs[i] /= sum;
        }
    }
    
    // Step 5: Sample from multinomial distribution
    float random = lcg_random(&rng_state);
    float cumulative = 0.0f;
    uint32_t sampled_idx = vocab_size - 1; // Fallback to last token
    
    for (size_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (random <= cumulative) {
            sampled_idx = indices[i];
            break;
        }
    }
    
    // Write result
    output[0] = sampled_idx;
}

// Kernel launcher for different data types
#define MULTINOMIAL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME* logits, \
    uint32_t* output, \
    const size_t vocab_size, \
    const float temperature, \
    const uint32_t top_k, \
    const float top_p, \
    const uint64_t seed \
) { \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        multinomial_sampling_kernel<TYPENAME>( \
            logits, output, vocab_size, temperature, top_k, top_p, seed \
        ); \
    } \
}

MULTINOMIAL_OP(float, multinomial_f32)
MULTINOMIAL_OP(double, multinomial_f64)
MULTINOMIAL_OP(__half, multinomial_f16)

#if __CUDA_ARCH__ >= 800
MULTINOMIAL_OP(__nv_bfloat16, multinomial_bf16)
#endif
