#include <metal_stdlib>

using namespace metal;

// Simple LCG random number generator (GPU-friendly)
float lcg_random(thread uint64_t* seed) {
    *seed = (*seed * 1664525ULL + 1013904223ULL) & 0xFFFFFFFFULL;
    return (*seed % 1000000) / 1000000.0f;
}

// Metal multinomial sampling kernel
template<typename T>
kernel void multinomial_sampling(
    constant T* logits [[buffer(0)]],           // Input logits
    device uint32_t* output [[buffer(1)]],      // Output token ID
    constant uint32_t& vocab_size [[buffer(2)]], // Vocabulary size
    constant float& temperature [[buffer(3)]],  // Temperature
    constant uint32_t& top_k [[buffer(4)]],     // Top-k (0 = disabled)
    constant float& top_p [[buffer(5)]],        // Top-p (0.0 = disabled)
    constant uint64_t& seed [[buffer(6)]],      // Random seed
    uint tid [[thread_position_in_grid]]
) {
    // Only first thread does the work
    if (tid != 0) return;
    
    // Allocate threadgroup memory for processing
    // Note: In practice, you'd use threadgroup memory for better performance
    // For now, we'll use a simplified approach
    
    // Allocate dynamic arrays (Metal doesn't support variable-length arrays in kernels directly,
    // so in practice this needs to be handled differently)
    // This is a simplified version - production code would need proper memory management
    
    thread float scores[32768];  // Max vocab size supported
    thread float probs[32768];
    thread uint32_t indices[32768];
    
    if (vocab_size > 32768) {
        output[0] = 0;
        return;
    }
    
    // Initialize random seed
    thread uint64_t rng_state = seed;
    
    // Step 1: Apply temperature scaling
    for (uint32_t i = 0; i < vocab_size; i++) {
        scores[i] = float(logits[i]) / temperature;
        indices[i] = i;
    }
    
    // Step 2: Apply top-k filtering if enabled
    uint32_t active_size = vocab_size;
    if (top_k > 0 && top_k < vocab_size) {
        // Partial sort to get top-k elements (selection algorithm)
        for (uint32_t k = 0; k < top_k; k++) {
            float max_val = -INFINITY;
            uint32_t max_idx = k;
            for (uint32_t i = k; i < vocab_size; i++) {
                if (scores[i] > max_val) {
                    max_val = scores[i];
                    max_idx = i;
                }
            }
            // Swap
            float tmp_score = scores[k];
            scores[k] = scores[max_idx];
            scores[max_idx] = tmp_score;
            
            uint32_t tmp_idx = indices[k];
            indices[k] = indices[max_idx];
            indices[max_idx] = tmp_idx;
        }
        
        // Mask out elements beyond top-k
        for (uint32_t i = top_k; i < vocab_size; i++) {
            scores[i] = -INFINITY;
        }
        active_size = top_k;
    }
    
    // Step 3: Convert to probabilities (softmax)
    float max_score = -INFINITY;
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        if (scores[i] != -INFINITY) {
            probs[i] = exp(scores[i] - max_score);
            sum += probs[i];
        } else {
            probs[i] = 0.0f;
        }
    }
    
    // Normalize
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Step 4: Apply top-p (nucleus) sampling if enabled
    if (top_p > 0.0f && top_p < 1.0f) {
        // Sort by probability (descending) - bubble sort for simplicity
        for (uint32_t i = 0; i < active_size - 1; i++) {
            for (uint32_t j = i + 1; j < active_size; j++) {
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
        uint32_t cutoff = active_size;
        for (uint32_t i = 0; i < active_size; i++) {
            cumulative += probs[i];
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Zero out probabilities beyond cutoff
        for (uint32_t i = cutoff; i < vocab_size; i++) {
            probs[i] = 0.0f;
        }
        
        // Renormalize
        sum = 0.0f;
        for (uint32_t i = 0; i < cutoff; i++) {
            sum += probs[i];
        }
        for (uint32_t i = 0; i < cutoff; i++) {
            probs[i] /= sum;
        }
    }
    
    // Step 5: Sample from multinomial distribution
    float random = lcg_random(&rng_state);
    float cumulative = 0.0f;
    uint32_t sampled_idx = vocab_size - 1; // Fallback
    
    for (uint32_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (random <= cumulative) {
            sampled_idx = indices[i];
            break;
        }
    }
    
    // Write result
    output[0] = sampled_idx;
}

// Explicit instantiations for different types
#define INSTANTIATE_MULTINOMIAL(TYPENAME, SUFFIX) \
template [[host_name("multinomial_" #SUFFIX)]] \
kernel void multinomial_sampling<TYPENAME>( \
    constant TYPENAME* logits [[buffer(0)]], \
    device uint32_t* output [[buffer(1)]], \
    constant uint32_t& vocab_size [[buffer(2)]], \
    constant float& temperature [[buffer(3)]], \
    constant uint32_t& top_k [[buffer(4)]], \
    constant float& top_p [[buffer(5)]], \
    constant uint64_t& seed [[buffer(6)]], \
    uint tid [[thread_position_in_grid]] \
);

INSTANTIATE_MULTINOMIAL(float, f32)
INSTANTIATE_MULTINOMIAL(half, f16)
// Note: Metal doesn't have native bfloat16, would need custom implementation
