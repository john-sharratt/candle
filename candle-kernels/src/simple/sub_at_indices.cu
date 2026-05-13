#include "cuda_utils.cuh"
#include<stdint.h>

// Optimized kernel for f32 - uses atomic operations for thread safety
extern "C" __global__ void sub_at_indices_f32(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        atomicAdd(&data[token_id], -value);
    }
}

// Optimized kernel for f16
extern "C" __global__ void sub_at_indices_f16(
    __half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        const __half val = __float2half(value);
        
        // Use native half atomics on newer GPUs
        atomicAdd(&data[token_id], __hneg(val));
    }
}

// Optimized kernel for bf16
extern "C" __global__ void sub_at_indices_bf16(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Use native bfloat16 atomics on Ampere and newer
        const __nv_bfloat16 val = __float2bfloat16(value);
        atomicAdd(&data[token_id], __hneg(val));
    }
}

// Optimized kernel for f64
extern "C" __global__ void sub_at_indices_f64(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        atomicAdd(&data[token_id], -value);
    }
}
