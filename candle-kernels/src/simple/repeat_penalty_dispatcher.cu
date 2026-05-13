// =============================================================================
// REPEAT PENALTY OPERATIONS DISPATCHER
// =============================================================================
// Provides a single extern "C" entry point that dispatches to the appropriate
// typed kernel based on dtype parameter for repeat penalty operations.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include <cuda_bf16.h>

// =============================================================================
// DType enum values (matching Rust RepeatPenaltyDType):
// 0=f32, 1=f64, 2=f16, 3=bf16
// =============================================================================

#define NUM_REPEAT_PENALTY_DTYPES 4

// =============================================================================
// Forward declarations of repeat penalty kernels
// =============================================================================
// Signature: (data, indices, num_indices, penalty)

extern "C" __global__ void repeat_penalty_f32(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
);

extern "C" __global__ void repeat_penalty_f64(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double penalty
);

extern "C" __global__ void repeat_penalty_f16(
    __half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
);

extern "C" __global__ void repeat_penalty_bf16(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
);

// =============================================================================
// Calculate grid and block dimensions for repeat penalty
// =============================================================================
static inline void get_repeat_penalty_launch_config(
    size_t num_indices,
    dim3& grid,
    dim3& block
) {
    const int threads_per_block = 256;
    int num_blocks = (num_indices + threads_per_block - 1) / threads_per_block;
    block = dim3(threads_per_block, 1, 1);
    grid = dim3(num_blocks, 1, 1);
}

// =============================================================================
// Repeat Penalty Dispatcher
// =============================================================================
// Applies repeat penalty to logits at specified token indices.
// For positive logits: divide by penalty (reduces probability)
// For negative/zero logits: multiply by penalty (reduces probability)
//
// This is used to discourage the model from repeating tokens that have
// already appeared in the generated sequence.
//
// Parameters:
//   dtype: data type enum (0=f32, 1=f64, 2=f16, 3=bf16)
//   data: mutable logits array
//   indices: token indices to penalize (previously generated tokens)
//   num_indices: number of indices to penalize
//   penalty: penalty value (typically > 1.0, e.g., 1.1 to 1.5)
//   stream: CUDA stream

extern "C" void run_repeat_penalty(
    int32_t dtype,
    void* data,
    const uint32_t* indices,
    size_t num_indices,
    double penalty,  // Use double to handle both f32 and f64 precision
    cudaStream_t stream
) {
    if (num_indices == 0) {
        return;  // Nothing to penalize
    }

    dim3 grid, block;
    get_repeat_penalty_launch_config(num_indices, grid, block);

    switch (dtype) {
        case 0: // f32
            repeat_penalty_f32<<<grid, block, 0, stream>>>(
                (float*)data, indices, num_indices, (float)penalty);
            break;
        case 1: // f64
            repeat_penalty_f64<<<grid, block, 0, stream>>>(
                (double*)data, indices, num_indices, penalty);
            break;
        case 2: // f16
            repeat_penalty_f16<<<grid, block, 0, stream>>>(
                (__half*)data, indices, num_indices, (float)penalty);
            break;
        case 3: // bf16
            repeat_penalty_bf16<<<grid, block, 0, stream>>>(
                (__nv_bfloat16*)data, indices, num_indices, (float)penalty);
            break;
        default:
            // Unsupported dtype - do nothing
            break;
    }
}

// =============================================================================
// Batch Repeat Penalty Dispatcher
// =============================================================================
// Applies repeat penalty to multiple batches of logits.
// Each batch has its own set of indices to penalize.
//
// Parameters:
//   dtype: data type enum (0=f32, 1=f64, 2=f16, 3=bf16)
//   data: mutable logits array, shape [batch_size, vocab_size]
//   indices: token indices to penalize for each batch, shape [batch_size, max_indices]
//   num_indices_per_batch: actual number of indices for each batch, shape [batch_size]
//   batch_size: number of batches
//   vocab_size: vocabulary size (stride between batches in data)
//   max_indices: maximum number of indices per batch (stride between batches in indices)
//   penalty: penalty value (typically > 1.0)
//   stream: CUDA stream

extern "C" void run_repeat_penalty_batch(
    int32_t dtype,
    void* data,
    const uint32_t* indices,
    const size_t* num_indices_per_batch,
    size_t batch_size,
    size_t vocab_size,
    size_t max_indices,
    double penalty,
    cudaStream_t stream
) {
    // Process each batch
    // Note: For better performance with many batches, a batched kernel could be implemented
    for (size_t b = 0; b < batch_size; ++b) {
        size_t num_indices = num_indices_per_batch[b];
        if (num_indices == 0) {
            continue;  // Skip empty batches
        }

        // Calculate pointers for this batch
        const uint32_t* batch_indices = indices + b * max_indices;
        
        // Calculate data pointer offset based on dtype size
        void* batch_data;
        switch (dtype) {
            case 0: // f32
                batch_data = (float*)data + b * vocab_size;
                break;
            case 1: // f64
                batch_data = (double*)data + b * vocab_size;
                break;
            case 2: // f16
                batch_data = (__half*)data + b * vocab_size;
                break;
            case 3: // bf16
                batch_data = (__nv_bfloat16*)data + b * vocab_size;
                break;
            default:
                continue;  // Unsupported dtype
        }

        run_repeat_penalty(dtype, batch_data, batch_indices, num_indices, penalty, stream);
    }
}
