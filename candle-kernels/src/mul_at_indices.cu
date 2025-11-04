#include "cuda_utils.cuh"
#include<stdint.h>

// Optimized kernel for f32 - uses atomic operations for thread safety
extern "C" __global__ void mul_at_indices_f32(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Atomic multiply using CAS (Compare-And-Swap)
        unsigned int* address_as_uint = (unsigned int*)&data[token_id];
        unsigned int old = *address_as_uint;
        unsigned int assumed;
        
        do {
            assumed = old;
            float old_float = __uint_as_float(assumed);
            float new_float = old_float * value;
            old = atomicCAS(address_as_uint, assumed, __float_as_uint(new_float));
        } while (assumed != old);
    }
}

// Optimized kernel for f16
extern "C" __global__ void mul_at_indices_f16(
    __half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Fallback using CAS (Compare-And-Swap) for thread safety
        unsigned short int* address_as_us = (unsigned short int*)&data[token_id];
        unsigned short int old = *address_as_us;
        unsigned short int assumed;
        
        do {
            assumed = old;
            __half old_half = __ushort_as_half(assumed);
            __half new_half = __float2half(__half2float(old_half) * value);
            old = atomicCAS(address_as_us, assumed, __half_as_ushort(new_half));
        } while (assumed != old);
    }
}

// Optimized kernel for bf16
extern "C" __global__ void mul_at_indices_bf16(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Fallback using CAS for thread safety
        unsigned short int* address_as_us = (unsigned short int*)&data[token_id];
        unsigned short int old = *address_as_us;
        unsigned short int assumed;
        
        do {
            assumed = old;
            __nv_bfloat16 old_bf16 = *reinterpret_cast<__nv_bfloat16*>(&assumed);
            float old_float = __bfloat162float(old_bf16);
            __nv_bfloat16 new_bf16 = __float2bfloat16(old_float * value);
            old = atomicCAS(address_as_us, assumed, *reinterpret_cast<unsigned short int*>(&new_bf16));
        } while (assumed != old);
    }
}

// Optimized kernel for f64
extern "C" __global__ void mul_at_indices_f64(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_indices) {
        const uint32_t token_id = indices[idx];
        
        // Atomic multiply using CAS (Compare-And-Swap)
        unsigned long long int* address_as_ull = (unsigned long long int*)&data[token_id];
        unsigned long long int old = *address_as_ull;
        unsigned long long int assumed;
        
        do {
            assumed = old;
            double old_double = __longlong_as_double(assumed);
            double new_double = old_double * value;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_double));
        } while (assumed != old);
    }
}
