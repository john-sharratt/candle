// =============================================================================
// FILL OPERATIONS DISPATCHER
// =============================================================================
// Provides extern "C" entry points that dispatch to the appropriate
// typed fill kernels based on dtype parameter.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// =============================================================================
// Grid/block configuration
// =============================================================================
#define BLOCK_SIZE 256

static inline unsigned int grid_size(size_t numel) {
    return (unsigned int)((numel + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// =============================================================================
// DType enum values (matching Rust FillDType):
// 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64
// =============================================================================

#define NUM_DTYPES 8

// =============================================================================
// Forward declarations: fill kernels (all are __global__)
// Signature: (buf, value, numel)
// =============================================================================
extern "C" __global__ void fill_u8(uint8_t*, uint8_t, size_t);
extern "C" __global__ void fill_u32(uint32_t*, uint32_t, size_t);
extern "C" __global__ void fill_i64(int64_t*, int64_t, size_t);
extern "C" __global__ void fill_f32(float*, float, size_t);
extern "C" __global__ void fill_f64(double*, double, size_t);
extern "C" __global__ void fill_f16(__half*, __half, size_t);
extern "C" __global__ void fill_bf16(__nv_bfloat16*, __nv_bfloat16, size_t);
extern "C" __global__ void fill_f8_e4m3(__nv_fp8_e4m3*, __nv_fp8_e4m3, size_t);

// =============================================================================
// Forward declarations: copy2d kernels (all are __global__)
// Signature: (src, dst, d1, d2, src_s, dst_s)
// =============================================================================
extern "C" __global__ void copy2d_u8(const uint8_t*, uint8_t*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_u32(const uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_i64(const int64_t*, int64_t*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_f32(const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_f64(const double*, double*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_f16(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_bf16(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);
extern "C" __global__ void copy2d_f8_e4m3(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);

// =============================================================================
// Forward declarations: const_set kernels (strided fill, all are __global__)
// Signature: (numel, num_dims, info, value, out)
// =============================================================================
extern "C" __global__ void const_set_u8(size_t, size_t, const size_t*, uint8_t, uint8_t*);
extern "C" __global__ void const_set_u32(size_t, size_t, const size_t*, uint32_t, uint32_t*);
extern "C" __global__ void const_set_i64(size_t, size_t, const size_t*, int64_t, int64_t*);
extern "C" __global__ void const_set_f32(size_t, size_t, const size_t*, float, float*);
extern "C" __global__ void const_set_f64(size_t, size_t, const size_t*, double, double*);
extern "C" __global__ void const_set_f16(size_t, size_t, const size_t*, __half, void*);
extern "C" __global__ void const_set_bf16(size_t, size_t, const size_t*, __nv_bfloat16, void*);
extern "C" __global__ void const_set_f8_e4m3(size_t, size_t, const size_t*, __nv_fp8_e4m3, void*);

// =============================================================================
// Dispatcher for fill operation
// =============================================================================
// Fills a contiguous buffer with a constant value.
// The value is passed as a uint64_t and reinterpreted based on dtype.
//
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64
//
// For floating point types, the value bits should be:
// - f32: lower 32 bits contain the float value
// - f64: all 64 bits contain the double value
// - f16: lower 16 bits contain the half value
// - bf16: lower 16 bits contain the bfloat16 value
// - f8_e4m3: lower 8 bits contain the fp8 value

extern "C" void run_fill_op(
    int32_t dtype,
    void* buf,
    uint64_t value_bits,
    size_t numel
) {
    unsigned int grid = grid_size(numel);
    
    switch (dtype) {
        case 0: // f32
            {
                uint32_t bits = (uint32_t)value_bits;
                float value;
                memcpy(&value, &bits, sizeof(float));
                fill_f32<<<grid, BLOCK_SIZE>>>((float*)buf, value, numel);
            }
            break;
        case 1: // f64
            {
                double value;
                memcpy(&value, &value_bits, sizeof(double));
                fill_f64<<<grid, BLOCK_SIZE>>>((double*)buf, value, numel);
            }
            break;
        case 2: // f16
            {
                uint16_t bits = (uint16_t)value_bits;
                __half value;
                memcpy(&value, &bits, sizeof(__half));
                fill_f16<<<grid, BLOCK_SIZE>>>((__half*)buf, value, numel);
            }
            break;
        case 3: // bf16
            {
                uint16_t bits = (uint16_t)value_bits;
                __nv_bfloat16 value;
                memcpy(&value, &bits, sizeof(__nv_bfloat16));
                fill_bf16<<<grid, BLOCK_SIZE>>>((__nv_bfloat16*)buf, value, numel);
            }
            break;
        case 4: // f8_e4m3
            {
                uint8_t bits = (uint8_t)value_bits;
                __nv_fp8_e4m3 value;
                memcpy(&value, &bits, sizeof(__nv_fp8_e4m3));
                fill_f8_e4m3<<<grid, BLOCK_SIZE>>>((__nv_fp8_e4m3*)buf, value, numel);
            }
            break;
        case 5: // u8
            fill_u8<<<grid, BLOCK_SIZE>>>((uint8_t*)buf, (uint8_t)value_bits, numel);
            break;
        case 6: // u32
            fill_u32<<<grid, BLOCK_SIZE>>>((uint32_t*)buf, (uint32_t)value_bits, numel);
            break;
        case 7: // i64
            fill_i64<<<grid, BLOCK_SIZE>>>((int64_t*)buf, (int64_t)value_bits, numel);
            break;
    }
}

// =============================================================================
// Dispatcher for copy2d operation
// =============================================================================
// Copies a 2D slice from src to dst with different strides.
//
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64

extern "C" void run_copy2d_op(
    int32_t dtype,
    const void* src,
    void* dst,
    uint32_t d1,
    uint32_t d2,
    uint32_t src_s,
    uint32_t dst_s
) {
    size_t numel = (size_t)d1 * (size_t)d2;
    unsigned int grid = grid_size(numel);
    
    switch (dtype) {
        case 0: // f32
            copy2d_f32<<<grid, BLOCK_SIZE>>>((const float*)src, (float*)dst, d1, d2, src_s, dst_s);
            break;
        case 1: // f64
            copy2d_f64<<<grid, BLOCK_SIZE>>>((const double*)src, (double*)dst, d1, d2, src_s, dst_s);
            break;
        case 2: // f16
            copy2d_f16<<<grid, BLOCK_SIZE>>>(src, dst, d1, d2, src_s, dst_s);
            break;
        case 3: // bf16
            copy2d_bf16<<<grid, BLOCK_SIZE>>>(src, dst, d1, d2, src_s, dst_s);
            break;
        case 4: // f8_e4m3
            copy2d_f8_e4m3<<<grid, BLOCK_SIZE>>>(src, dst, d1, d2, src_s, dst_s);
            break;
        case 5: // u8
            copy2d_u8<<<grid, BLOCK_SIZE>>>((const uint8_t*)src, (uint8_t*)dst, d1, d2, src_s, dst_s);
            break;
        case 6: // u32
            copy2d_u32<<<grid, BLOCK_SIZE>>>((const uint32_t*)src, (uint32_t*)dst, d1, d2, src_s, dst_s);
            break;
        case 7: // i64
            copy2d_i64<<<grid, BLOCK_SIZE>>>((const int64_t*)src, (int64_t*)dst, d1, d2, src_s, dst_s);
            break;
    }
}

// =============================================================================
// Dispatcher for const_set operation (strided fill)
// =============================================================================
// Sets all elements in a strided tensor to a constant value.
// The value is passed as a uint64_t and reinterpreted based on dtype.
//
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64

extern "C" void run_const_set_op(
    int32_t dtype,
    size_t numel,
    size_t num_dims,
    const size_t* info,
    uint64_t value_bits,
    void* out
) {
    unsigned int grid = grid_size(numel);
    
    switch (dtype) {
        case 0: // f32
            {
                uint32_t bits = (uint32_t)value_bits;
                float value;
                memcpy(&value, &bits, sizeof(float));
                const_set_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, value, (float*)out);
            }
            break;
        case 1: // f64
            {
                double value;
                memcpy(&value, &value_bits, sizeof(double));
                const_set_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, value, (double*)out);
            }
            break;
        case 2: // f16
            {
                uint16_t bits = (uint16_t)value_bits;
                __half value;
                memcpy(&value, &bits, sizeof(__half));
                const_set_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, value, out);
            }
            break;
        case 3: // bf16
            {
                uint16_t bits = (uint16_t)value_bits;
                __nv_bfloat16 value;
                memcpy(&value, &bits, sizeof(__nv_bfloat16));
                const_set_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, value, out);
            }
            break;
        case 4: // f8_e4m3
            {
                uint8_t bits = (uint8_t)value_bits;
                __nv_fp8_e4m3 value;
                memcpy(&value, &bits, sizeof(__nv_fp8_e4m3));
                const_set_f8_e4m3<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, value, out);
            }
            break;
        case 5: // u8
            const_set_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (uint8_t)value_bits, (uint8_t*)out);
            break;
        case 6: // u32
            const_set_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (uint32_t)value_bits, (uint32_t*)out);
            break;
        case 7: // i64
            const_set_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (int64_t)value_bits, (int64_t*)out);
            break;
    }
}
