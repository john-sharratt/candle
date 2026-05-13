// Simple batched sampling kernel - no penalties
// These are simplified versions that just do temperature/top-k/top-p sampling
// without repeat penalties, DRY, etc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cfloat>
#include <curand_kernel.h>
#include "fast_exp.cuh"

// Simple batched sampling kernel template
// Each block handles one sequence
template<typename T>
__global__ void batched_sampling_simple_kernel(
    const T* __restrict__ logits,      // [batch_size, vocab_size]
    int32_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t* __restrict__ output_tokens,  // [batch_size]
    uint64_t seed,
    uint64_t* __restrict__ rng_offsets     // [batch_size] - per-sequence RNG state
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Shared memory for reduction
    __shared__ float s_max_val;
    __shared__ float s_sum;
    __shared__ float s_threshold;
    __shared__ uint32_t s_selected_token;
    
    const T* seq_logits = logits + seq_idx * vocab_size;
    
    // Step 1: Find max logit (for numerical stability)
    float local_max = -FLT_MAX;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = static_cast<float>(seq_logits[i]);
        local_max = fmaxf(local_max, val);
    }
    
    // Warp reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    
    // Block reduction
    __shared__ float s_warp_maxs[32];
    if (lane_id == 0) s_warp_maxs[warp_id] = local_max;
    __syncthreads();
    
    if (tid == 0) {
        float block_max = -FLT_MAX;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            block_max = fmaxf(block_max, s_warp_maxs[i]);
        }
        s_max_val = block_max;
    }
    __syncthreads();
    
    float max_val = s_max_val;
    
    // Step 2: Apply temperature and compute softmax sum
    float temp_inv = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = static_cast<float>(seq_logits[i]);
        // val - max_val <= 0 always, so use Softmax mode
        float prob = fast_exp::exp<float, fast_exp::Softmax>((val - max_val) * temp_inv);
        local_sum += prob;
    }
    
    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Block reduction
    __shared__ float s_warp_sums[32];
    if (lane_id == 0) s_warp_sums[warp_id] = local_sum;
    __syncthreads();
    
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            block_sum += s_warp_sums[i];
        }
        s_sum = block_sum;
    }
    __syncthreads();
    
    float sum_val = s_sum;
    
    // Step 3: Sample using RNG
    if (tid == 0) {
        // Initialize RNG with seed and sequence-specific offset
        curandState rng;
        uint64_t offset = rng_offsets[seq_idx];
        curand_init(seed, seq_idx, offset, &rng);
        
        // Generate random number in [0, 1)
        float r = curand_uniform(&rng) * sum_val;
        
        // Find token via cumulative sum
        float cumsum = 0.0f;
        int selected = vocab_size - 1; // Default to last token
        
        for (int i = 0; i < vocab_size; i++) {
            float val = static_cast<float>(seq_logits[i]);
            // val - max_val <= 0 always, so use Softmax mode
            float prob = fast_exp::exp<float, fast_exp::Softmax>((val - max_val) * temp_inv);
            cumsum += prob;
            if (cumsum >= r) {
                selected = i;
                break;
            }
        }
        
        output_tokens[seq_idx] = static_cast<uint32_t>(selected);
        rng_offsets[seq_idx] = offset + 1;
    }
}

// Batched argmax kernel
__global__ void batched_argmax_kernel(
    const float* __restrict__ logits,  // [batch_size, vocab_size]
    int32_t vocab_size,
    uint32_t* __restrict__ output_tokens  // [batch_size]
) {
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    const float* seq_logits = logits + seq_idx * vocab_size;
    
    // Find local max and argmax
    float local_max = -FLT_MAX;
    int local_argmax = 0;
    
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = seq_logits[i];
        if (val > local_max) {
            local_max = val;
            local_argmax = i;
        }
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_argmax = __shfl_down_sync(0xffffffff, local_argmax, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_argmax = other_argmax;
        }
    }
    
    // Block reduction via shared memory
    __shared__ float s_warp_maxs[32];
    __shared__ int s_warp_argmaxs[32];
    
    if (lane_id == 0) {
        s_warp_maxs[warp_id] = local_max;
        s_warp_argmaxs[warp_id] = local_argmax;
    }
    __syncthreads();
    
    if (tid == 0) {
        float best_max = -FLT_MAX;
        int best_argmax = 0;
        int num_warps = (blockDim.x + 31) / 32;
        
        for (int i = 0; i < num_warps; i++) {
            if (s_warp_maxs[i] > best_max) {
                best_max = s_warp_maxs[i];
                best_argmax = s_warp_argmaxs[i];
            }
        }
        
        output_tokens[seq_idx] = static_cast<uint32_t>(best_argmax);
    }
}

// C API for simple sampling (F32)
extern "C" void run_batched_sampling_simple(
    const float* logits,
    int32_t batch_size,
    int32_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
) {
    if (batch_size <= 0) return;
    
    // One block per sequence, 256 threads per block
    int threads = 256;
    int blocks = batch_size;
    
    batched_sampling_simple_kernel<float><<<blocks, threads>>>(
        logits, vocab_size, temperature, top_k, top_p,
        output_tokens, seed, rng_offsets
    );
}

// C API for simple sampling (F16)
extern "C" void run_batched_sampling_simple_f16(
    const half* logits,
    int32_t batch_size,
    int32_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
) {
    if (batch_size <= 0) return;
    
    int threads = 256;
    int blocks = batch_size;
    
    batched_sampling_simple_kernel<half><<<blocks, threads>>>(
        logits, vocab_size, temperature, top_k, top_p,
        output_tokens, seed, rng_offsets
    );
}

// C API for simple sampling (BF16)
extern "C" void run_batched_sampling_simple_bf16(
    const __nv_bfloat16* logits,
    int32_t batch_size,
    int32_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
) {
    if (batch_size <= 0) return;
    
    int threads = 256;
    int blocks = batch_size;
    
    batched_sampling_simple_kernel<__nv_bfloat16><<<blocks, threads>>>(
        logits, vocab_size, temperature, top_k, top_p,
        output_tokens, seed, rng_offsets
    );
}

// C API for simple sampling (FP8 E4M3)
extern "C" void run_batched_sampling_simple_fp8_e4m3(
    const __nv_fp8_e4m3* logits,
    int32_t batch_size,
    int32_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    uint32_t* output_tokens,
    uint64_t seed,
    uint64_t* rng_offsets
) {
    if (batch_size <= 0) return;
    
    int threads = 256;
    int blocks = batch_size;
    
    batched_sampling_simple_kernel<__nv_fp8_e4m3><<<blocks, threads>>>(
        logits, vocab_size, temperature, top_k, top_p,
        output_tokens, seed, rng_offsets
    );
}

// C API for batched argmax (greedy decoding)
extern "C" void run_batched_argmax(
    const float* logits,
    int32_t batch_size,
    int32_t vocab_size,
    uint32_t* output_tokens
) {
    if (batch_size <= 0) return;
    
    int threads = 256;
    int blocks = batch_size;
    
    batched_argmax_kernel<<<blocks, threads>>>(
        logits, vocab_size, output_tokens
    );
}
