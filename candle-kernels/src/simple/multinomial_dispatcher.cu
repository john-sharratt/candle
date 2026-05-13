// =============================================================================
// MULTINOMIAL OPERATIONS DISPATCHER
// =============================================================================
// Provides a single extern "C" entry point that dispatches to the appropriate
// typed kernel based on dtype parameter for multinomial sampling.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cuda_bf16.h>

// =============================================================================
// DType enum values (matching Rust MultinomialDType):
// 0=f32, 1=f64, 2=f16, 3=bf16
// =============================================================================

#define NUM_MULTINOMIAL_DTYPES 4

// =============================================================================
// Forward declarations of multinomial kernels
// =============================================================================
// Signature: (logits, output, workspace, vocab_size, temperature, top_k, top_p, seed)

extern "C" __global__ void optimized_multinomial_f32(
    const float* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
);

extern "C" __global__ void optimized_multinomial_f64(
    const double* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
);

extern "C" __global__ void optimized_multinomial_f16(
    const __half* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
);

extern "C" __global__ void optimized_multinomial_bf16(
    const __nv_bfloat16* logits,
    uint32_t* output,
    float* workspace,
    const size_t vocab_size,
    const float temperature,
    const uint32_t top_k,
    const float top_p,
    const uint64_t seed
);

// =============================================================================
// Multinomial Sampling Dispatcher
// =============================================================================
// GPU-accelerated multinomial sampling with full top-k and top-p support.
//
// Parameters:
//   dtype: data type enum (0=f32, 1=f64, 2=f16, 3=bf16)
//   logits: input logits array
//   output: output sampled index (single uint32_t)
//   workspace: workspace for intermediate computations (size: vocab_size * 8 bytes)
//   vocab_size: size of the vocabulary
//   temperature: temperature for sampling (higher = more random)
//   top_k: top-k filtering (0 = disabled, limits to k most likely tokens)
//   top_p: top-p (nucleus) filtering (1.0 = disabled, limits to tokens with cumulative prob >= top_p)
//   seed: random seed for sampling
//   num_threads: number of threads for parallel reductions (typically 256 or 512)
//   shared_mem_size: shared memory size in bytes for parallel reductions
//   stream: CUDA stream

extern "C" void run_multinomial(
    int32_t dtype,
    const void* logits,
    uint32_t* output,
    float* workspace,
    size_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    uint64_t seed,
    int num_threads,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    // Use single block with multiple threads for parallel reductions
    dim3 block(num_threads, 1, 1);
    dim3 grid(1, 1, 1);

    switch (dtype) {
        case 0: // f32
            optimized_multinomial_f32<<<grid, block, shared_mem_size, stream>>>(
                (const float*)logits, output, workspace, vocab_size,
                temperature, top_k, top_p, seed);
            break;
        case 1: // f64
            optimized_multinomial_f64<<<grid, block, shared_mem_size, stream>>>(
                (const double*)logits, output, workspace, vocab_size,
                temperature, top_k, top_p, seed);
            break;
        case 2: // f16
            optimized_multinomial_f16<<<grid, block, shared_mem_size, stream>>>(
                (const __half*)logits, output, workspace, vocab_size,
                temperature, top_k, top_p, seed);
            break;
        case 3: // bf16
            optimized_multinomial_bf16<<<grid, block, shared_mem_size, stream>>>(
                (const __nv_bfloat16*)logits, output, workspace, vocab_size,
                temperature, top_k, top_p, seed);
            break;
        default:
            // Unsupported dtype - do nothing
            break;
    }
}

// =============================================================================
// Simple Multinomial Sampling (no top-k/top-p)
// =============================================================================
// Convenience dispatcher for basic multinomial sampling without filtering.
// Uses default parameters: top_k=0, top_p=1.0 (both disabled)

extern "C" void run_multinomial_simple(
    int32_t dtype,
    const void* logits,
    uint32_t* output,
    float* workspace,
    size_t vocab_size,
    float temperature,
    uint64_t seed,
    int num_threads,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    run_multinomial(
        dtype, logits, output, workspace, vocab_size,
        temperature, 0, 1.0f, seed,  // top_k=0, top_p=1.0 (disabled)
        num_threads, shared_mem_size, stream
    );
}

// =============================================================================
// Workspace Size Calculator
// =============================================================================
// Returns the required workspace size in bytes for multinomial sampling.
// Workspace is used for storing (probability, index) pairs during sorting.

extern "C" size_t get_multinomial_workspace_size(size_t vocab_size) {
    // ProbIndex struct is 8 bytes (4 bytes float prob + 4 bytes uint32 index)
    return vocab_size * 8;
}
