// =============================================================================
// SCATTER OPERATIONS DISPATCHER
// =============================================================================
// Provides a single extern "C" entry point that dispatches to the appropriate
// typed kernel based on operation type and dtype parameters.
//
// Scatter operations: data[indices[i]] op= value
// - Add: data[idx] += value
// - Sub: data[idx] -= value  
// - Mul: data[idx] *= value
// - Div: data[idx] /= value
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// =============================================================================
// Operation and dtype enums (must match Rust side)
// =============================================================================

enum ScatterOp {
    SCATTER_ADD = 0,
    SCATTER_SUB = 1,
    SCATTER_MUL = 2,
    SCATTER_DIV = 3,
};

enum ScatterDType {
    SCATTER_F32 = 0,
    SCATTER_F64 = 1,
    SCATTER_F16 = 2,
    SCATTER_BF16 = 3,
};

// =============================================================================
// Forward declarations of all kernel variants
// =============================================================================

// Note: add_at_indices has stride parameter, others don't
// We normalize by adding stride to all operations in the dispatcher

// add_at_indices variants (from add_at_indices.cu)
extern "C" __global__ void add_at_indices_f32(float*, const uint32_t*, size_t, float, size_t);
extern "C" __global__ void add_at_indices_f64(double*, const uint32_t*, size_t, double, size_t);
extern "C" __global__ void add_at_indices_f16(__half*, const uint32_t*, size_t, __half, size_t);
extern "C" __global__ void add_at_indices_bf16(__nv_bfloat16*, const uint32_t*, size_t, __nv_bfloat16, size_t);

// sub_at_indices variants (from sub_at_indices.cu) - no stride parameter
extern "C" __global__ void sub_at_indices_f32(float*, const uint32_t*, size_t, float);
extern "C" __global__ void sub_at_indices_f64(double*, const uint32_t*, size_t, double);
extern "C" __global__ void sub_at_indices_f16(__half*, const uint32_t*, size_t, float);
extern "C" __global__ void sub_at_indices_bf16(__nv_bfloat16*, const uint32_t*, size_t, float);

// mul_at_indices variants (from mul_at_indices.cu) - no stride parameter
extern "C" __global__ void mul_at_indices_f32(float*, const uint32_t*, size_t, float);
extern "C" __global__ void mul_at_indices_f64(double*, const uint32_t*, size_t, double);
extern "C" __global__ void mul_at_indices_f16(__half*, const uint32_t*, size_t, float);
extern "C" __global__ void mul_at_indices_bf16(__nv_bfloat16*, const uint32_t*, size_t, float);

// div_at_indices variants (from div_at_indices.cu) - no stride parameter
extern "C" __global__ void div_at_indices_f32(float*, const uint32_t*, size_t, float);
extern "C" __global__ void div_at_indices_f64(double*, const uint32_t*, size_t, double);
extern "C" __global__ void div_at_indices_f16(__half*, const uint32_t*, size_t, float);
extern "C" __global__ void div_at_indices_bf16(__nv_bfloat16*, const uint32_t*, size_t, float);

// =============================================================================
// Dispatcher implementation
// =============================================================================

/// Unified dispatcher for scatter operations (add, sub, mul, div) at indices.
///
/// For add operations, the stride parameter is used.
/// For sub/mul/div operations, stride is currently ignored (those kernels don't support it).
///
/// @param op       Operation type (0=add, 1=sub, 2=mul, 3=div)
/// @param dtype    Data type (0=f32, 1=f64, 2=f16, 3=bf16)
/// @param data     Pointer to tensor data (will be modified in-place)
/// @param indices  Pointer to indices array (u32)
/// @param num_indices Number of indices
/// @param value_f32 Value as f32 (used for f32/f16/bf16)
/// @param value_f64 Value as f64 (used for f64)
/// @param stride   Stride between elements (only used by add operation)
extern "C" void run_scatter_op_at_indices(
    int op,
    int dtype,
    void* data,
    const uint32_t* indices,
    size_t num_indices,
    float value_f32,
    double value_f64,
    size_t stride
) {
    if (num_indices == 0) return;
    
    const int threads = 256;
    const int blocks = (num_indices + threads - 1) / threads;
    
    switch (op) {
        case SCATTER_ADD:
            switch (dtype) {
                case SCATTER_F32:
                    add_at_indices_f32<<<blocks, threads>>>(
                        (float*)data, indices, num_indices, value_f32, stride);
                    break;
                case SCATTER_F64:
                    add_at_indices_f64<<<blocks, threads>>>(
                        (double*)data, indices, num_indices, value_f64, stride);
                    break;
                case SCATTER_F16: {
                    __half value_f16 = __float2half(value_f32);
                    add_at_indices_f16<<<blocks, threads>>>(
                        (__half*)data, indices, num_indices, value_f16, stride);
                    break;
                }
                case SCATTER_BF16: {
                    __nv_bfloat16 value_bf16 = __float2bfloat16(value_f32);
                    add_at_indices_bf16<<<blocks, threads>>>(
                        (__nv_bfloat16*)data, indices, num_indices, value_bf16, stride);
                    break;
                }
            }
            break;
            
        case SCATTER_SUB:
            switch (dtype) {
                case SCATTER_F32:
                    sub_at_indices_f32<<<blocks, threads>>>(
                        (float*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_F64:
                    sub_at_indices_f64<<<blocks, threads>>>(
                        (double*)data, indices, num_indices, value_f64);
                    break;
                case SCATTER_F16:
                    sub_at_indices_f16<<<blocks, threads>>>(
                        (__half*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_BF16:
                    sub_at_indices_bf16<<<blocks, threads>>>(
                        (__nv_bfloat16*)data, indices, num_indices, value_f32);
                    break;
            }
            break;
            
        case SCATTER_MUL:
            switch (dtype) {
                case SCATTER_F32:
                    mul_at_indices_f32<<<blocks, threads>>>(
                        (float*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_F64:
                    mul_at_indices_f64<<<blocks, threads>>>(
                        (double*)data, indices, num_indices, value_f64);
                    break;
                case SCATTER_F16:
                    mul_at_indices_f16<<<blocks, threads>>>(
                        (__half*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_BF16:
                    mul_at_indices_bf16<<<blocks, threads>>>(
                        (__nv_bfloat16*)data, indices, num_indices, value_f32);
                    break;
            }
            break;
            
        case SCATTER_DIV:
            switch (dtype) {
                case SCATTER_F32:
                    div_at_indices_f32<<<blocks, threads>>>(
                        (float*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_F64:
                    div_at_indices_f64<<<blocks, threads>>>(
                        (double*)data, indices, num_indices, value_f64);
                    break;
                case SCATTER_F16:
                    div_at_indices_f16<<<blocks, threads>>>(
                        (__half*)data, indices, num_indices, value_f32);
                    break;
                case SCATTER_BF16:
                    div_at_indices_bf16<<<blocks, threads>>>(
                        (__nv_bfloat16*)data, indices, num_indices, value_f32);
                    break;
            }
            break;
    }
}

// =============================================================================
// Separate dispatcher for sub_at_indices_with_values (different signature)
// =============================================================================

// Forward declarations for sub_at_indices_with_values (from sub_at_indices_with_values.cu)
extern "C" __global__ void sub_at_indices_with_values_f32(float*, const uint32_t*, const float*, size_t);
extern "C" __global__ void sub_at_indices_with_values_f64(double*, const uint32_t*, const double*, size_t);
extern "C" __global__ void sub_at_indices_with_values_f16(__half*, const uint32_t*, const float*, size_t);
extern "C" __global__ void sub_at_indices_with_values_bf16(__nv_bfloat16*, const uint32_t*, const float*, size_t);

/// Dispatcher for sub_at_indices_with_values operation.
/// Each index gets its own value: data[indices[i]] -= values[i]
///
/// @param dtype    Data type (0=f32, 1=f64, 2=f16, 3=bf16)
/// @param data     Pointer to tensor data (will be modified in-place)
/// @param indices  Pointer to indices array (u32)
/// @param values   Pointer to values array (f32 for f16/bf16, native type for f32/f64)
/// @param num_indices Number of indices
extern "C" void run_sub_at_indices_with_values(
    int dtype,
    void* data,
    const uint32_t* indices,
    const void* values,
    size_t num_indices
) {
    if (num_indices == 0) return;
    
    const int threads = 256;
    const int blocks = (num_indices + threads - 1) / threads;
    
    switch (dtype) {
        case SCATTER_F32:
            sub_at_indices_with_values_f32<<<blocks, threads>>>(
                (float*)data, indices, (const float*)values, num_indices);
            break;
        case SCATTER_F64:
            sub_at_indices_with_values_f64<<<blocks, threads>>>(
                (double*)data, indices, (const double*)values, num_indices);
            break;
        case SCATTER_F16:
            // f16 kernel expects float values array (converts internally)
            sub_at_indices_with_values_f16<<<blocks, threads>>>(
                (__half*)data, indices, (const float*)values, num_indices);
            break;
        case SCATTER_BF16:
            // bf16 kernel expects float values array (converts internally)
            sub_at_indices_with_values_bf16<<<blocks, threads>>>(
                (__nv_bfloat16*)data, indices, (const float*)values, num_indices);
            break;
    }
}
