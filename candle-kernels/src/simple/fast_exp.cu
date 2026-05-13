// =============================================================================
// fast_exp.cu - CUDA kernels for batch fast_exp operations
// =============================================================================
// Exposes the fast_exp library functions to Rust via extern "C" entry points.
// Processes arrays of values through the high-performance software exp.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "../fast_exp.cuh"

constexpr int BLOCK_SIZE = 256;

inline int grid_size(size_t numel, int block_size = BLOCK_SIZE) {
    return (numel + block_size - 1) / block_size;
}

// =============================================================================
// BATCH EXP KERNELS
// =============================================================================

// Generic mode (safe for any input) - Cubic precision
__global__ void fast_exp_batch_f32_generic_high(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Generic, fast_exp::High>(inp[i]);
    }
}

// Generic mode - Quadratic precision
__global__ void fast_exp_batch_f32_generic_medium(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Generic, fast_exp::Medium>(inp[i]);
    }
}

// Generic mode - Linear precision
__global__ void fast_exp_batch_f32_generic_low(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Generic, fast_exp::Low>(inp[i]);
    }
}

// Softmax mode (assumes x <= 0) - Cubic precision
__global__ void fast_exp_batch_f32_softmax_high(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Softmax, fast_exp::High>(inp[i]);
    }
}

// Softmax mode - Quadratic precision
__global__ void fast_exp_batch_f32_softmax_medium(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Softmax, fast_exp::Medium>(inp[i]);
    }
}

// Softmax mode - Linear precision
__global__ void fast_exp_batch_f32_softmax_low(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<float, fast_exp::Softmax, fast_exp::Low>(inp[i]);
    }
}

// =============================================================================
// BATCH ACTIVATION KERNELS (F32)
// =============================================================================

__global__ void fast_sigmoid_batch_f32(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::sigmoid<float>(inp[i]);
    }
}

__global__ void fast_silu_batch_f32(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::silu<float>(inp[i]);
    }
}

__global__ void fast_gelu_batch_f32(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::gelu<float>(inp[i]);
    }
}

// =============================================================================
// FP16 BATCH KERNELS
// =============================================================================

__global__ void fast_exp_batch_f16_softmax_high(
    const __half* __restrict__ inp,
    __half* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<__half, fast_exp::Softmax, fast_exp::High>(inp[i]);
    }
}

__global__ void fast_exp_batch_f16_softmax_medium(
    const __half* __restrict__ inp,
    __half* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<__half, fast_exp::Softmax, fast_exp::Medium>(inp[i]);
    }
}

__global__ void fast_sigmoid_batch_f16(
    const __half* __restrict__ inp,
    __half* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::sigmoid<__half>(inp[i]);
    }
}

// =============================================================================
// BF16 BATCH KERNELS
// =============================================================================

__global__ void fast_exp_batch_bf16_softmax_low(
    const __nv_bfloat16* __restrict__ inp,
    __nv_bfloat16* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::exp<nv_bfloat16, fast_exp::Softmax, fast_exp::Low>(inp[i]);
    }
}

__global__ void fast_sigmoid_batch_bf16(
    const __nv_bfloat16* __restrict__ inp,
    __nv_bfloat16* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = fast_exp::sigmoid<nv_bfloat16>(inp[i]);
    }
}

// =============================================================================
// REFERENCE EXP KERNEL (uses hardware __expf for comparison)
// =============================================================================

__global__ void reference_exp_batch_f32(
    const float* __restrict__ inp,
    float* __restrict__ out,
    size_t numel
) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        out[i] = __expf(inp[i]);
    }
}

// =============================================================================
// EXTERN "C" WRAPPER FUNCTIONS
// =============================================================================

extern "C" {

// ---- Dispatcher for fast_exp batch ----
// mode: 0=Generic, 1=Softmax
// precision: 0=High (cubic), 1=Medium (quadratic), 2=Low (linear)
// dtype: 0=f32, 1=f16, 2=bf16
void run_fast_exp_batch(
    int32_t mode,
    int32_t precision,
    int32_t dtype,
    const void* inp,
    void* out,
    size_t numel
) {
    int grid = grid_size(numel);
    
    if (dtype == 0) { // f32
        if (mode == 0) { // Generic
            switch (precision) {
                case 0: fast_exp_batch_f32_generic_high<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
                case 1: fast_exp_batch_f32_generic_medium<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
                case 2: fast_exp_batch_f32_generic_low<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
            }
        } else { // Softmax
            switch (precision) {
                case 0: fast_exp_batch_f32_softmax_high<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
                case 1: fast_exp_batch_f32_softmax_medium<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
                case 2: fast_exp_batch_f32_softmax_low<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
            }
        }
    }
    else if (dtype == 1) { // f16
        if (mode == 1) { // Softmax only for f16
            switch (precision) {
                case 0: fast_exp_batch_f16_softmax_high<<<grid, BLOCK_SIZE>>>((const __half*)inp, (__half*)out, numel); break;
                case 1: fast_exp_batch_f16_softmax_medium<<<grid, BLOCK_SIZE>>>((const __half*)inp, (__half*)out, numel); break;
            }
        }
    }
    else if (dtype == 2) { // bf16
        if (mode == 1) { // Softmax only for bf16
            fast_exp_batch_bf16_softmax_low<<<grid, BLOCK_SIZE>>>((const __nv_bfloat16*)inp, (__nv_bfloat16*)out, numel);
        }
    }
}

// ---- Dispatcher for activation functions ----
// op: 0=sigmoid, 1=silu, 2=gelu
// dtype: 0=f32, 1=f16, 2=bf16
void run_fast_activation_batch(
    int32_t op,
    int32_t dtype,
    const void* inp,
    void* out,
    size_t numel
) {
    int grid = grid_size(numel);
    
    if (dtype == 0) { // f32
        switch (op) {
            case 0: fast_sigmoid_batch_f32<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
            case 1: fast_silu_batch_f32<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
            case 2: fast_gelu_batch_f32<<<grid, BLOCK_SIZE>>>((const float*)inp, (float*)out, numel); break;
        }
    }
    else if (dtype == 1) { // f16
        if (op == 0) {
            fast_sigmoid_batch_f16<<<grid, BLOCK_SIZE>>>((const __half*)inp, (__half*)out, numel);
        }
    }
    else if (dtype == 2) { // bf16
        if (op == 0) {
            fast_sigmoid_batch_bf16<<<grid, BLOCK_SIZE>>>((const __nv_bfloat16*)inp, (__nv_bfloat16*)out, numel);
        }
    }
}

// ---- Reference exp for testing ----
void run_reference_exp_batch(
    const float* inp,
    float* out,
    size_t numel
) {
    int grid = grid_size(numel);
    reference_exp_batch_f32<<<grid, BLOCK_SIZE>>>(inp, out, numel);
}

} // extern "C"
