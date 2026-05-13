// =============================================================================
// REDUCE OPERATIONS DISPATCHER
// =============================================================================
// Provides extern "C" entry points that dispatch to the appropriate
// typed reduce kernels based on operation type and dtype parameters.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// =============================================================================
// Forward declarations of all kernel variants
// =============================================================================

// Signature type for fast reduce ops (sum, min, max) - same src/dst type
using fast_reduce_fn_t = void (*)(
    size_t src_numel, size_t el_to_sum_per_block,
    size_t num_dims, const size_t* info,
    const void* src, void* dst
);

// Signature type for fast arg reduce ops (argmin, argmax) - dst is always u32
using fast_arg_reduce_fn_t = void (*)(
    size_t src_numel, size_t el_to_sum_per_block,
    size_t num_dims, const size_t* info,
    const void* src, uint32_t* dst
);

// Signature type for sum op (with atomicAdd)
using sum_fn_t = void (*)(
    size_t numel, size_t num_dims, size_t num_sum_dims,
    const size_t* info, const void* inp, void* out
);

// Signature type for softmax
using softmax_fn_t = void (*)(
    const void* src, void* dst, int n_cols
);

// Signature type for rmsnorm
using rmsnorm_fn_t = void (*)(
    const void* src, void* dst, const void* alpha,
    int n_cols, int block_size, float eps
);

// Signature type for layernorm
using layernorm_fn_t = void (*)(
    const void* src, void* dst, const void* alpha, const void* beta,
    int n_cols, int block_size, float eps
);

// Signature type for rope_i (interleaved)
using rope_i_fn_t = void (*)(
    const void* src, const void* cos, const void* sin, void* dst,
    uint32_t bh, uint32_t td, uint32_t stride_b
);

// Signature type for rope (non-interleaved)
using rope_fn_t = void (*)(
    const void* src, const void* cos, const void* sin, void* dst,
    uint32_t bh, uint32_t td, uint32_t d, uint32_t stride_b
);

// Signature type for rope_thd
using rope_thd_fn_t = void (*)(
    const void* src, const void* cos, const void* sin, void* dst,
    uint32_t b, uint32_t t, uint32_t h, uint32_t d, uint32_t stride_b
);

// =============================================================================
// FAST_SUM kernels
// =============================================================================
extern "C" void fast_sum_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" void fast_sum_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" void fast_sum_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void fast_sum_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" void fast_sum_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" void fast_sum_f16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void fast_sum_bf16(size_t, size_t, size_t, const size_t*, const void*, void*);

// =============================================================================
// FAST_MIN kernels
// =============================================================================
extern "C" void fast_min_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" void fast_min_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" void fast_min_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void fast_min_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" void fast_min_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" void fast_min_f16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void fast_min_bf16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void fast_min_f8_e4m3(size_t, size_t, size_t, const size_t*, const void*, void*);

// =============================================================================
// FAST_MAX kernels
// =============================================================================
extern "C" void fast_max_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" void fast_max_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" void fast_max_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void fast_max_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" void fast_max_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" void fast_max_f16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void fast_max_bf16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void fast_max_f8_e4m3(size_t, size_t, size_t, const size_t*, const void*, void*);

// =============================================================================
// FAST_ARGMIN kernels
// =============================================================================
extern "C" void fast_argmin_f32(size_t, size_t, size_t, const size_t*, const float*, uint32_t*);
extern "C" void fast_argmin_f64(size_t, size_t, size_t, const size_t*, const double*, uint32_t*);
extern "C" void fast_argmin_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void fast_argmin_i64(size_t, size_t, size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" void fast_argmin_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" void fast_argmin_f16(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);
extern "C" void fast_argmin_bf16(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);
extern "C" void fast_argmin_f8_e4m3(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);

// =============================================================================
// FAST_ARGMAX kernels
// =============================================================================
extern "C" void fast_argmax_f32(size_t, size_t, size_t, const size_t*, const float*, uint32_t*);
extern "C" void fast_argmax_f64(size_t, size_t, size_t, const size_t*, const double*, uint32_t*);
extern "C" void fast_argmax_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void fast_argmax_i64(size_t, size_t, size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" void fast_argmax_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" void fast_argmax_f16(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);
extern "C" void fast_argmax_bf16(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);
extern "C" void fast_argmax_f8_e4m3(size_t, size_t, size_t, const size_t*, const void*, uint32_t*);

// =============================================================================
// SUM kernels (with atomicAdd)
// =============================================================================
extern "C" void sum_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" void sum_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" void sum_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" void sum_f16(size_t, size_t, size_t, const size_t*, const void*, void*);
extern "C" void sum_bf16(size_t, size_t, size_t, const size_t*, const void*, void*);

// =============================================================================
// SOFTMAX kernels
// =============================================================================
extern "C" __global__ void softmax_f32(const float*, float*, int);
extern "C" __global__ void softmax_f64(const double*, double*, int);
extern "C" __global__ void softmax_f16(const void*, void*, int);
extern "C" __global__ void softmax_bf16(const void*, void*, int);
extern "C" __global__ void softmax_f8_e4m3(const void*, void*, int);

// =============================================================================
// RMSNORM kernels
// =============================================================================
extern "C" __global__ void rmsnorm_f32(const float*, float*, const float*, int, int, float);
extern "C" __global__ void rmsnorm_f64(const double*, double*, const double*, int, int, float);
extern "C" __global__ void rmsnorm_f16(const void*, void*, const void*, int, int, float);
extern "C" __global__ void rmsnorm_bf16(const void*, void*, const void*, int, int, float);
extern "C" __global__ void rmsnorm_f8_e4m3(const void*, void*, const void*, int, int, float);

// =============================================================================
// LAYERNORM kernels
// =============================================================================
extern "C" __global__ void layernorm_f32(const float*, float*, const float*, const float*, int, int, float);
extern "C" __global__ void layernorm_f64(const double*, double*, const double*, const double*, int, int, float);
extern "C" __global__ void layernorm_f16(const void*, void*, const void*, const void*, int, int, float);
extern "C" __global__ void layernorm_bf16(const void*, void*, const void*, const void*, int, int, float);
extern "C" __global__ void layernorm_f8_e4m3(const void*, void*, const void*, const void*, int, int, float);

// =============================================================================
// ROPE_I kernels (interleaved)
// =============================================================================
extern "C" __global__ void rope_i_f32(const float*, const float*, const float*, float*, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_i_f64(const double*, const double*, const double*, double*, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_i_f16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_i_bf16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_i_f8_e4m3(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t);

// =============================================================================
// ROPE kernels (non-interleaved)
// =============================================================================
extern "C" __global__ void rope_f32(const float*, const float*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_f64(const double*, const double*, const double*, double*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_f16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_bf16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_f8_e4m3(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);

// =============================================================================
// ROPE_THD kernels
// =============================================================================
extern "C" __global__ void rope_thd_f32(const float*, const float*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_thd_f64(const double*, const double*, const double*, double*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_thd_f16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_thd_bf16(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void rope_thd_f8_e4m3(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

// =============================================================================
// Operation and DType enum values
// =============================================================================
// FastReduceOp: 0=sum, 1=min, 2=max
// FastArgReduceOp: 0=argmin, 1=argmax
// DType for fast reduce: 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3
// DType for sum/softmax/norm: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3 (no sum for f8_e4m3)
// DType for rope: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3

#define NUM_SUM_DTYPES 5
#define NUM_FLOAT_DTYPES 5

// NOTE: run_fast_reduce_op and run_fast_arg_reduce_op are defined in api.cu

// =============================================================================
// Dispatcher for sum operation (with atomicAdd, multi-dimensional reduce)
// =============================================================================
extern "C" void run_sum_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32
    size_t numel,
    size_t num_dims,
    size_t num_sum_dims,
    const size_t* info,
    const void* inp,
    void* out
) {
    static sum_fn_t kernels[NUM_SUM_DTYPES] = {
        (sum_fn_t)sum_f32,
        (sum_fn_t)sum_f64,
        (sum_fn_t)sum_f16,
        (sum_fn_t)sum_bf16,
        (sum_fn_t)sum_u32,
    };

    if (dtype >= 0 && dtype < NUM_SUM_DTYPES) {
        sum_fn_t fn = kernels[dtype];
        if (fn != nullptr) {
            fn(numel, num_dims, num_sum_dims, info, inp, out);
        }
    }
}

// =============================================================================
// Dispatcher for softmax operation
// =============================================================================
extern "C" void run_softmax_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    void* dst,
    int n_rows,
    int n_cols
) {
    // Launch configuration: one block per row, 32 threads per block
    dim3 grid(n_rows, 1, 1);
    dim3 block(1, 32, 1);
    
    switch (dtype) {
        case 0: // F32
            softmax_f32<<<grid, block>>>((const float*)src, (float*)dst, n_cols);
            break;
        case 1: // F64
            softmax_f64<<<grid, block>>>((const double*)src, (double*)dst, n_cols);
            break;
        case 2: // F16
            softmax_f16<<<grid, block>>>((const __half*)src, (__half*)dst, n_cols);
            break;
        case 3: // BF16
            softmax_bf16<<<grid, block>>>((const __nv_bfloat16*)src, (__nv_bfloat16*)dst, n_cols);
            break;
        case 4: // F8E4M3
            softmax_f8_e4m3<<<grid, block>>>((const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst, n_cols);
            break;
    }
}

// =============================================================================
// Dispatcher for rmsnorm operation
// =============================================================================

// Maximum columns we can cache in shared memory (must match reduce.cu)
#define MAX_CACHED_COLS 8192

extern "C" void run_rmsnorm_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    void* dst,
    const void* alpha,
    int n_rows,
    int n_cols,
    float eps
) {
    // Launch configuration: one block per row
    // Cap block size to 1024 (CUDA max) - kernel handles striding over columns
    // Minimum 32 threads to guarantee a full warp for correct __shfl_xor_sync reductions
    int block_size = n_cols < 1024 ? n_cols : 1024;
    if (block_size < 32) block_size = 32;
    dim3 grid(n_rows, 1, 1);
    dim3 block(block_size, 1, 1);
    
    // Calculate shared memory size for x_cache
    // Only allocate if ncols fits in cache, otherwise 0
    size_t shared_mem_size = (n_cols <= MAX_CACHED_COLS) ? (n_cols * sizeof(float)) : 0;
    
    switch (dtype) {
        case 0: // F32
            rmsnorm_f32<<<grid, block, shared_mem_size>>>((const float*)src, (float*)dst, (const float*)alpha, n_cols, block_size, eps);
            break;
        case 1: // F64
            rmsnorm_f64<<<grid, block, shared_mem_size>>>((const double*)src, (double*)dst, (const double*)alpha, n_cols, block_size, eps);
            break;
        case 2: // F16
            rmsnorm_f16<<<grid, block, shared_mem_size>>>((const __half*)src, (__half*)dst, (const __half*)alpha, n_cols, block_size, eps);
            break;
        case 3: // BF16
            rmsnorm_bf16<<<grid, block, shared_mem_size>>>((const __nv_bfloat16*)src, (__nv_bfloat16*)dst, (const __nv_bfloat16*)alpha, n_cols, block_size, eps);
            break;
        case 4: // F8E4M3
            rmsnorm_f8_e4m3<<<grid, block, shared_mem_size>>>((const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst, (const __nv_fp8_e4m3*)alpha, n_cols, block_size, eps);
            break;
    }
}

// =============================================================================
// Dispatcher for layernorm operation
// =============================================================================
extern "C" void run_layernorm_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    void* dst,
    const void* alpha,
    const void* beta,
    int n_rows,
    int n_cols,
    float eps
) {
    // Launch configuration: one block per row
    // Cap block size to 1024 (CUDA max) - kernel handles striding over columns
    // Minimum 32 threads to guarantee a full warp for correct __shfl_xor_sync reductions
    int block_size = n_cols < 1024 ? n_cols : 1024;
    if (block_size < 32) block_size = 32;
    dim3 grid(n_rows, 1, 1);
    dim3 block(block_size, 1, 1);
    
    switch (dtype) {
        case 0: // F32
            layernorm_f32<<<grid, block>>>((const float*)src, (float*)dst, (const float*)alpha, (const float*)beta, n_cols, block_size, eps);
            break;
        case 1: // F64
            layernorm_f64<<<grid, block>>>((const double*)src, (double*)dst, (const double*)alpha, (const double*)beta, n_cols, block_size, eps);
            break;
        case 2: // F16
            layernorm_f16<<<grid, block>>>((const __half*)src, (__half*)dst, (const __half*)alpha, (const __half*)beta, n_cols, block_size, eps);
            break;
        case 3: // BF16
            layernorm_bf16<<<grid, block>>>((const __nv_bfloat16*)src, (__nv_bfloat16*)dst, (const __nv_bfloat16*)alpha, (const __nv_bfloat16*)beta, n_cols, block_size, eps);
            break;
        case 4: // F8E4M3
            layernorm_f8_e4m3<<<grid, block>>>((const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst, (const __nv_fp8_e4m3*)alpha, (const __nv_fp8_e4m3*)beta, n_cols, block_size, eps);
            break;
    }
}

// =============================================================================
// Dispatcher for rope_i operation (interleaved)
// =============================================================================
extern "C" void run_rope_i_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    const void* cos,
    const void* sin,
    void* dst,
    uint32_t bh,
    uint32_t td,
    uint32_t stride_b
) {
    // Kernel needs bh * td / 2 total threads
    // (the kernel uses blockIdx.x * blockDim.x + threadIdx.x as idx,
    //  and processes 2 elements per thread)
    const uint32_t total_threads = (bh * td) / 2;
    const int block_dim = 256;
    const int grid_dim = (total_threads + block_dim - 1) / block_dim;
    
    switch (dtype) {
        case 0: // F32
            rope_i_f32<<<grid_dim, block_dim>>>((const float*)src, (const float*)cos, (const float*)sin, (float*)dst, bh, td, stride_b);
            break;
        case 1: // F64
            rope_i_f64<<<grid_dim, block_dim>>>((const double*)src, (const double*)cos, (const double*)sin, (double*)dst, bh, td, stride_b);
            break;
        case 2: // F16
            rope_i_f16<<<grid_dim, block_dim>>>((const __half*)src, (const __half*)cos, (const __half*)sin, (__half*)dst, bh, td, stride_b);
            break;
        case 3: // BF16
            rope_i_bf16<<<grid_dim, block_dim>>>((const __nv_bfloat16*)src, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, (__nv_bfloat16*)dst, bh, td, stride_b);
            break;
        case 4: // F8E4M3
            rope_i_f8_e4m3<<<grid_dim, block_dim>>>((const __nv_fp8_e4m3*)src, (const __nv_fp8_e4m3*)cos, (const __nv_fp8_e4m3*)sin, (__nv_fp8_e4m3*)dst, bh, td, stride_b);
            break;
    }
}

// =============================================================================
// Dispatcher for rope operation (non-interleaved)
// =============================================================================
extern "C" void run_rope_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    const void* cos,
    const void* sin,
    void* dst,
    uint32_t bh,
    uint32_t td,
    uint32_t d,
    uint32_t stride_b
) {
    // Kernel needs bh * td / 2 total threads
    // (the kernel uses blockIdx.x * blockDim.x + threadIdx.x as idx,
    //  and processes 2 elements per thread)
    const uint32_t total_threads = (bh * td) / 2;
    const int block_dim = 256;
    const int grid_dim = (total_threads + block_dim - 1) / block_dim;
    
    switch (dtype) {
        case 0: // F32
            rope_f32<<<grid_dim, block_dim>>>((const float*)src, (const float*)cos, (const float*)sin, (float*)dst, bh, td, d, stride_b);
            break;
        case 1: // F64
            rope_f64<<<grid_dim, block_dim>>>((const double*)src, (const double*)cos, (const double*)sin, (double*)dst, bh, td, d, stride_b);
            break;
        case 2: // F16
            rope_f16<<<grid_dim, block_dim>>>((const __half*)src, (const __half*)cos, (const __half*)sin, (__half*)dst, bh, td, d, stride_b);
            break;
        case 3: // BF16
            rope_bf16<<<grid_dim, block_dim>>>((const __nv_bfloat16*)src, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, (__nv_bfloat16*)dst, bh, td, d, stride_b);
            break;
        case 4: // F8E4M3
            rope_f8_e4m3<<<grid_dim, block_dim>>>((const __nv_fp8_e4m3*)src, (const __nv_fp8_e4m3*)cos, (const __nv_fp8_e4m3*)sin, (__nv_fp8_e4m3*)dst, bh, td, d, stride_b);
            break;
    }
}

// =============================================================================
// Dispatcher for rope_thd operation
// =============================================================================
extern "C" void run_rope_thd_op(
    int32_t dtype,   // 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3
    const void* src,
    const void* cos,
    const void* sin,
    void* dst,
    uint32_t b,
    uint32_t t,
    uint32_t h,
    uint32_t d,
    uint32_t stride_b
) {
    // Compute grid/block dimensions
    const uint32_t el_count = b * t * h * d;
    const int block_dim = 256;
    const int grid_dim = (el_count + block_dim - 1) / block_dim;
    
    switch (dtype) {
        case 0: // F32
            rope_thd_f32<<<grid_dim, block_dim>>>((const float*)src, (const float*)cos, (const float*)sin, (float*)dst, b, t, h, d, stride_b);
            break;
        case 1: // F64
            rope_thd_f64<<<grid_dim, block_dim>>>((const double*)src, (const double*)cos, (const double*)sin, (double*)dst, b, t, h, d, stride_b);
            break;
        case 2: // F16
            rope_thd_f16<<<grid_dim, block_dim>>>((const __half*)src, (const __half*)cos, (const __half*)sin, (__half*)dst, b, t, h, d, stride_b);
            break;
        case 3: // BF16
            rope_thd_bf16<<<grid_dim, block_dim>>>((const __nv_bfloat16*)src, (const __nv_bfloat16*)cos, (const __nv_bfloat16*)sin, (__nv_bfloat16*)dst, b, t, h, d, stride_b);
            break;
        case 4: // F8E4M3
            rope_thd_f8_e4m3<<<grid_dim, block_dim>>>((const __nv_fp8_e4m3*)src, (const __nv_fp8_e4m3*)cos, (const __nv_fp8_e4m3*)sin, (__nv_fp8_e4m3*)dst, b, t, h, d, stride_b);
            break;
    }
}

