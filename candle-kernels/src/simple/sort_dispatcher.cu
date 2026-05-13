// =============================================================================
// SORT OPERATIONS DISPATCHER
// =============================================================================
// Provides extern "C" entry points that dispatch to the appropriate
// typed kernel based on dtype parameter for sort/argsort operations.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// =============================================================================
// Forward declarations of all argsort kernel variants
// =============================================================================

// Argsort ascending kernels
extern "C" __global__ void asort_asc_f32(const float*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_f64(const double*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_u8(const uint8_t*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_u32(const uint32_t*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_i64(const int64_t*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_f16(const void*, uint32_t*, int, int);
extern "C" __global__ void asort_asc_bf16(const void*, uint32_t*, int, int);

// Argsort descending kernels
extern "C" __global__ void asort_desc_f32(const float*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_f64(const double*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_u8(const uint8_t*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_u32(const uint32_t*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_i64(const int64_t*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_f16(const void*, uint32_t*, int, int);
extern "C" __global__ void asort_desc_bf16(const void*, uint32_t*, int, int);

// =============================================================================
// DType enum values
// =============================================================================
// 0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32, 6=i64

#define NUM_SORT_DTYPES 7

// =============================================================================
// Argsort Ascending Dispatcher
// =============================================================================
// Returns sorted indices (argsort) in ascending order
// Parameters:
//   dtype: data type enum (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32, 6=i64)
//   x: input data pointer
//   dst: output indices pointer (uint32_t)
//   ncols: number of columns (elements per row to sort)
//   ncols_pad: padded column count (power of 2 for bitonic sort)
//   nrows: number of rows (batch dimension)
//   shared_mem_size: shared memory size in bytes
//   stream: CUDA stream

extern "C" void run_argsort_asc(
    int32_t dtype,
    const void* x,
    uint32_t* dst,
    int ncols,
    int ncols_pad,
    int nrows,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    dim3 block(ncols_pad, 1, 1);
    dim3 grid(nrows, 1, 1);

    switch (dtype) {
        case 0: // f32
            asort_asc_f32<<<grid, block, shared_mem_size, stream>>>(
                (const float*)x, dst, ncols, ncols_pad);
            break;
        case 1: // f64
            asort_asc_f64<<<grid, block, shared_mem_size, stream>>>(
                (const double*)x, dst, ncols, ncols_pad);
            break;
        case 2: // f16
            asort_asc_f16<<<grid, block, shared_mem_size, stream>>>(
                x, dst, ncols, ncols_pad);
            break;
        case 3: // bf16
            asort_asc_bf16<<<grid, block, shared_mem_size, stream>>>(
                x, dst, ncols, ncols_pad);
            break;
        case 4: // u8
            asort_asc_u8<<<grid, block, shared_mem_size, stream>>>(
                (const uint8_t*)x, dst, ncols, ncols_pad);
            break;
        case 5: // u32
            asort_asc_u32<<<grid, block, shared_mem_size, stream>>>(
                (const uint32_t*)x, dst, ncols, ncols_pad);
            break;
        case 6: // i64
            asort_asc_i64<<<grid, block, shared_mem_size, stream>>>(
                (const int64_t*)x, dst, ncols, ncols_pad);
            break;
    }
}

// =============================================================================
// Argsort Descending Dispatcher
// =============================================================================
// Returns sorted indices (argsort) in descending order

extern "C" void run_argsort_desc(
    int32_t dtype,
    const void* x,
    uint32_t* dst,
    int ncols,
    int ncols_pad,
    int nrows,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    dim3 block(ncols_pad, 1, 1);
    dim3 grid(nrows, 1, 1);

    switch (dtype) {
        case 0: // f32
            asort_desc_f32<<<grid, block, shared_mem_size, stream>>>(
                (const float*)x, dst, ncols, ncols_pad);
            break;
        case 1: // f64
            asort_desc_f64<<<grid, block, shared_mem_size, stream>>>(
                (const double*)x, dst, ncols, ncols_pad);
            break;
        case 2: // f16
            asort_desc_f16<<<grid, block, shared_mem_size, stream>>>(
                x, dst, ncols, ncols_pad);
            break;
        case 3: // bf16
            asort_desc_bf16<<<grid, block, shared_mem_size, stream>>>(
                x, dst, ncols, ncols_pad);
            break;
        case 4: // u8
            asort_desc_u8<<<grid, block, shared_mem_size, stream>>>(
                (const uint8_t*)x, dst, ncols, ncols_pad);
            break;
        case 5: // u32
            asort_desc_u32<<<grid, block, shared_mem_size, stream>>>(
                (const uint32_t*)x, dst, ncols, ncols_pad);
            break;
        case 6: // i64
            asort_desc_i64<<<grid, block, shared_mem_size, stream>>>(
                (const int64_t*)x, dst, ncols, ncols_pad);
            break;
    }
}

// =============================================================================
// Sort Ascending Dispatcher (sorts values in-place using argsort + gather)
// =============================================================================
// Note: The current kernels only support argsort (returning indices).
// A full sort would require either:
// 1. Using the argsort indices to gather values into a new array
// 2. Implementing separate sort kernels that sort values directly
//
// For now, these dispatchers provide the same functionality as argsort
// but are named for future expansion if direct value sorting is added.
//
// To perform a full sort of values:
// 1. Call run_argsort_asc/desc to get indices
// 2. Use the indices to gather values into sorted order

extern "C" void run_sort_asc(
    int32_t dtype,
    const void* x,
    uint32_t* dst,
    int ncols,
    int ncols_pad,
    int nrows,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    // Currently delegates to argsort - returns indices
    // Future: could implement direct value sorting
    run_argsort_asc(dtype, x, dst, ncols, ncols_pad, nrows, shared_mem_size, stream);
}

extern "C" void run_sort_desc(
    int32_t dtype,
    const void* x,
    uint32_t* dst,
    int ncols,
    int ncols_pad,
    int nrows,
    size_t shared_mem_size,
    cudaStream_t stream
) {
    // Currently delegates to argsort - returns indices
    // Future: could implement direct value sorting
    run_argsort_desc(dtype, x, dst, ncols, ncols_pad, nrows, shared_mem_size, stream);
}
