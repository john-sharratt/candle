#include "cuda_utils.cuh"
#include<stdint.h>

// Helper function for atomic float division using CAS
__device__ __forceinline__ float atomicDiv(float* address, float divisor) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        float old_float = __uint_as_float(assumed);
        float new_float = old_float / divisor;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(new_float));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

// Helper function for atomic float multiplication using CAS
__device__ __forceinline__ float atomicMul(float* address, float multiplier) {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        float old_float = __uint_as_float(assumed);
        float new_float = old_float * multiplier;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(new_float));
    } while (assumed != old);
    
    return __uint_as_float(old);
}

// Kernel for f32 - applies penalty based on sign of logit value
// Positive logits: divide by penalty (reduces probability)
// Negative/zero logits: multiply by penalty (reduces probability)
extern "C" __global__ void repeat_penalty_f32(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        float current_value = data[token_id];
        
        // Apply penalty based on sign
        if (current_value > 0.0f) {
            // Positive: divide by penalty
            atomicDiv(&data[token_id], penalty);
        } else {
            // Negative or zero: multiply by penalty
            atomicMul(&data[token_id], penalty);
        }
    }
}

// Kernel for f16
extern "C" __global__ void repeat_penalty_f16(
    __half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Use CAS (Compare-And-Swap) for thread safety
        unsigned short int* address_as_us = (unsigned short int*)&data[token_id];
        unsigned short int old = *address_as_us;
        unsigned short int assumed;
        
        do {
            assumed = old;
            __half old_half = __ushort_as_half(assumed);
            float old_float = __half2float(old_half);
            
            // Apply penalty based on sign
            float new_float;
            if (old_float > 0.0f) {
                new_float = old_float / penalty;
            } else {
                new_float = old_float * penalty;
            }
            
            __half new_half = __float2half(new_float);
            old = atomicCAS(address_as_us, assumed, __half_as_ushort(new_half));
        } while (assumed != old);
    }
}

// Kernel for bf16
extern "C" __global__ void repeat_penalty_bf16(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float penalty
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Use CAS for thread safety
        unsigned short int* address_as_us = (unsigned short int*)&data[token_id];
        unsigned short int old = *address_as_us;
        unsigned short int assumed;
        
        do {
            assumed = old;
            __nv_bfloat16 old_bf16 = *reinterpret_cast<__nv_bfloat16*>(&assumed);
            float old_float = __bfloat162float(old_bf16);
            
            // Apply penalty based on sign
            float new_float;
            if (old_float > 0.0f) {
                new_float = old_float / penalty;
            } else {
                new_float = old_float * penalty;
            }
            
            __nv_bfloat16 new_bf16 = __float2bfloat16(new_float);
            old = atomicCAS(address_as_us, assumed, *reinterpret_cast<unsigned short int*>(&new_bf16));
        } while (assumed != old);
    }
}

// Helper function for atomic double division using CAS
__device__ __forceinline__ double atomicDivDouble(double* address, double divisor) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        double old_double = __longlong_as_double(assumed);
        double new_double = old_double / divisor;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_double));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Helper function for atomic double multiplication using CAS
__device__ __forceinline__ double atomicMulDouble(double* address, double multiplier) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        double old_double = __longlong_as_double(assumed);
        double new_double = old_double * multiplier;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_double));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// Kernel for f64
extern "C" __global__ void repeat_penalty_f64(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double penalty
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        double current_value = data[token_id];
        
        // Apply penalty based on sign
        if (current_value > 0.0) {
            // Positive: divide by penalty
            atomicDivDouble(&data[token_id], penalty);
        } else {
            // Negative or zero: multiply by penalty
            atomicMulDouble(&data[token_id], penalty);
        }
    }
}
