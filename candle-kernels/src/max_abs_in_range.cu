#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <stdint.h>
#include <float.h>

// Warp-level reduction for max absolute value
template<typename T>
__device__ float warp_reduce_max_abs(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Block-level reduction for max absolute value
template<typename T>
__device__ float block_reduce_max_abs(float val) {
    __shared__ float shared[32]; // One per warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    // Warp-level reduction
    val = warp_reduce_max_abs<T>(val);
    
    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    // First warp reduces across warps
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_max_abs<T>(val);
    
    return val;
}

// Kernel for f32
extern "C" __global__ void max_abs_in_range_f32(
    const float *input,
    float *output,
    const size_t start,
    const size_t end,
    const size_t total_elems
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_size = end - start;
    
    float local_max = 0.0f;
    
    // Grid-stride loop
    for (size_t i = idx; i < range_size; i += blockDim.x * gridDim.x) {
        size_t global_idx = start + i;
        if (global_idx < end && global_idx < total_elems) {
            float val = fabsf(input[global_idx]);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Block-level reduction
    local_max = block_reduce_max_abs<float>(local_max);
    
    // First thread writes result
    if (threadIdx.x == 0) {
        atomicMax((int*)output, __float_as_int(local_max));
    }
}

// Kernel for f16
extern "C" __global__ void max_abs_in_range_f16(
    const __half *input,
    float *output,
    const size_t start,
    const size_t end,
    const size_t total_elems
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_size = end - start;
    
    float local_max = 0.0f;
    
    // Grid-stride loop
    for (size_t i = idx; i < range_size; i += blockDim.x * gridDim.x) {
        size_t global_idx = start + i;
        if (global_idx < end && global_idx < total_elems) {
            float val = fabsf(__half2float(input[global_idx]));
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Block-level reduction
    local_max = block_reduce_max_abs<__half>(local_max);
    
    // First thread writes result
    if (threadIdx.x == 0) {
        atomicMax((int*)output, __float_as_int(local_max));
    }
}

// Kernel for bf16
extern "C" __global__ void max_abs_in_range_bf16(
    const __nv_bfloat16 *input,
    float *output,
    const size_t start,
    const size_t end,
    const size_t total_elems
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_size = end - start;
    
    float local_max = 0.0f;
    
    // Grid-stride loop
    for (size_t i = idx; i < range_size; i += blockDim.x * gridDim.x) {
        size_t global_idx = start + i;
        if (global_idx < end && global_idx < total_elems) {
            float val = fabsf(__bfloat162float(input[global_idx]));
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Block-level reduction
    local_max = block_reduce_max_abs<__nv_bfloat16>(local_max);
    
    // First thread writes result
    if (threadIdx.x == 0) {
        atomicMax((int*)output, __float_as_int(local_max));
    }
}

// Kernel for f64
extern "C" __global__ void max_abs_in_range_f64(
    const double *input,
    float *output,
    const size_t start,
    const size_t end,
    const size_t total_elems
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t range_size = end - start;
    
    float local_max = 0.0f;
    
    // Grid-stride loop
    for (size_t i = idx; i < range_size; i += blockDim.x * gridDim.x) {
        size_t global_idx = start + i;
        if (global_idx < end && global_idx < total_elems) {
            float val = fabsf((float)input[global_idx]);
            local_max = fmaxf(local_max, val);
        }
    }
    
    // Block-level reduction
    local_max = block_reduce_max_abs<double>(local_max);
    
    // First thread writes result
    if (threadIdx.x == 0) {
        atomicMax((int*)output, __float_as_int(local_max));
    }
}
