#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include <stdint.h>

#define ARCH_SUPPORTS_BF16 1

// Default block size for simple kernels
#define BLOCK_SIZE 256

// Atomic add for float32
__device__ void atomic_add_f32(float* address, float val) {
    atomicAdd(address, val);
}

// Atomic add for float16 using compare-and-swap
__device__ void atomic_add_f16(half* address, half val) {
    unsigned int* address_as_uint = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        half old_val = ((size_t)address & 2) ? __ushort_as_half((unsigned short)(old >> 16))
                                              : __ushort_as_half((unsigned short)(old & 0xffff));
        half new_val = __hadd(old_val, val);
        unsigned short new_val_short = __half_as_ushort(new_val);
        
        unsigned int new_uint = ((size_t)address & 2) 
            ? (old & 0xffff) | (new_val_short << 16)
            : (old & 0xffff0000) | new_val_short;
            
        old = atomicCAS(address_as_uint, assumed, new_uint);
    } while (assumed != old);
}

// Atomic add for bfloat16 using compare-and-swap
__device__ void atomic_add_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    unsigned int* address_as_uint = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    
    do {
        assumed = old;
        __nv_bfloat16 old_val = ((size_t)address & 2) 
            ? __ushort_as_bfloat16((unsigned short)(old >> 16))
            : __ushort_as_bfloat16((unsigned short)(old & 0xffff));
        
        __nv_bfloat16 new_val = __hadd(old_val, val);
        unsigned short new_val_short = __bfloat16_as_ushort(new_val);
        
        unsigned int new_uint = ((size_t)address & 2)
            ? (old & 0xffff) | (new_val_short << 16)
            : (old & 0xffff0000) | new_val_short;
            
        old = atomicCAS(address_as_uint, assumed, new_uint);
    } while (assumed != old);
}

// Atomic add for float64 using compare-and-swap
__device__ void atomic_add_f64(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    
    do {
        assumed = old;
        double old_val = __longlong_as_double(old);
        double new_val = old_val + val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
}

// ============================================================================
// CUDA Kernels (device code)
// ============================================================================

__global__ void add_at_indices_f32_kernel(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value,
    const size_t stride
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        const size_t index = indices[idx];
        atomic_add_f32(&data[index * stride], value);
    }
}

__global__ void add_at_indices_f16_kernel(
    half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const half value,
    const size_t stride
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        const size_t index = indices[idx];
        atomic_add_f16(&data[index * stride], value);
    }
}

__global__ void add_at_indices_bf16_kernel(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const __nv_bfloat16 value,
    const size_t stride
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        const size_t index = indices[idx];
        atomic_add_bf16(&data[index * stride], value);
    }
}

__global__ void add_at_indices_f64_kernel(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double value,
    const size_t stride
) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_indices) {
        const size_t index = indices[idx];
        atomic_add_f64(&data[index * stride], value);
    }
}

// ============================================================================
// Host-side dispatcher functions (called from Rust via FFI)
// ============================================================================

extern "C" void add_at_indices_f32(
    float* data,
    const uint32_t* indices,
    const size_t num_indices,
    const float value,
    const size_t stride
) {
    if (num_indices == 0) return;
    
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (num_indices + block_size - 1) / block_size;
    
    add_at_indices_f32_kernel<<<num_blocks, block_size>>>(
        data, indices, num_indices, value, stride
    );
}

extern "C" void add_at_indices_f16(
    half* data,
    const uint32_t* indices,
    const size_t num_indices,
    const half value,
    const size_t stride
) {
    if (num_indices == 0) return;
    
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (num_indices + block_size - 1) / block_size;
    
    add_at_indices_f16_kernel<<<num_blocks, block_size>>>(
        data, indices, num_indices, value, stride
    );
}

extern "C" void add_at_indices_bf16(
    __nv_bfloat16* data,
    const uint32_t* indices,
    const size_t num_indices,
    const __nv_bfloat16 value,
    const size_t stride
) {
    if (num_indices == 0) return;
    
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (num_indices + block_size - 1) / block_size;
    
    add_at_indices_bf16_kernel<<<num_blocks, block_size>>>(
        data, indices, num_indices, value, stride
    );
}

extern "C" void add_at_indices_f64(
    double* data,
    const uint32_t* indices,
    const size_t num_indices,
    const double value,
    const size_t stride
) {
    if (num_indices == 0) return;
    
    const int block_size = BLOCK_SIZE;
    const int num_blocks = (num_indices + block_size - 1) / block_size;
    
    add_at_indices_f64_kernel<<<num_blocks, block_size>>>(
        data, indices, num_indices, value, stride
    );
}
