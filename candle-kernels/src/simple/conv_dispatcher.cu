// =============================================================================
// CONVOLUTION OPERATIONS DISPATCHER
// =============================================================================
// Provides single extern "C" entry points that dispatch to the appropriate
// typed kernel based on operation type and dtype parameters.
//
// Operations grouped by signature pattern:
// - conv1d: 1D convolution
// - conv2d: 2D convolution
// - conv_transpose1d: 1D transposed convolution
// - conv_transpose2d: 2D transposed convolution
// - im2col: Image to column transformation (2D)
// - im2col1d: Image to column transformation (1D)
// - col2im1d: Column to image transformation (1D)
// - avg_pool2d: 2D average pooling
// - max_pool2d: 2D max pooling
// - upsample_nearest2d: 2D nearest neighbor upsampling
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// =============================================================================
// Kernel launch configuration
// =============================================================================

constexpr int BLOCK_SIZE = 256;

inline int grid_size(size_t numel, int block_size = BLOCK_SIZE) {
    return (numel + block_size - 1) / block_size;
}

// =============================================================================
// Data type enum (must match Rust side)
// =============================================================================

enum ConvDType {
    CONV_F32 = 0,
    CONV_F64 = 1,
    CONV_F16 = 2,
    CONV_BF16 = 3,
    CONV_U8 = 4,
    CONV_U32 = 5,
};

// =============================================================================
// Forward declarations for conv1d kernels
// =============================================================================
// Signature: (src_numel, num_dims, stride, padding, dilation, info, src, kernel, dst)

extern "C" __global__ void conv1d_f32(size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void conv1d_f64(size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void conv1d_u8(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void conv1d_u32(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void conv1d_f16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void conv1d_bf16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for conv2d kernels
// =============================================================================
// Signature: (src_numel, w_out, h_out, stride, padding, dilation, info, src, kernel, dst)

extern "C" __global__ void conv2d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void conv2d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void conv2d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void conv2d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void conv2d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void conv2d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for conv_transpose1d kernels
// =============================================================================
// Signature: (src_numel, l_out, stride, padding, out_padding, dilation, info, src, kernel, dst)

extern "C" __global__ void conv_transpose1d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void conv_transpose1d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void conv_transpose1d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void conv_transpose1d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void conv_transpose1d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void conv_transpose1d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for conv_transpose2d kernels
// =============================================================================
// Signature: (src_numel, w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst)

extern "C" __global__ void conv_transpose2d_f32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void conv_transpose2d_f64(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void conv_transpose2d_u8(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void conv_transpose2d_u32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void conv_transpose2d_f16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void conv_transpose2d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for im2col kernels (2D)
// =============================================================================
// Signature: (dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst)

extern "C" __global__ void im2col_f32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void im2col_f64(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void im2col_u8(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void im2col_u32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void im2col_f16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void im2col_bf16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for im2col1d kernels
// =============================================================================
// Signature: (dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst)

extern "C" __global__ void im2col1d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void im2col1d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void im2col1d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void im2col1d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void im2col1d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void im2col1d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for col2im1d kernels
// =============================================================================
// Signature: (dst_el, l_out, l_in, c_out, k_size, stride, src, dst)

extern "C" __global__ void col2im1d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const float*, float*);
extern "C" __global__ void col2im1d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const double*, double*);
extern "C" __global__ void col2im1d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const uint8_t*, uint8_t*);
extern "C" __global__ void col2im1d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const uint32_t*, uint32_t*);
extern "C" __global__ void col2im1d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const __half*, __half*);
extern "C" __global__ void col2im1d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for avg_pool2d kernels
// =============================================================================
// Signature: (src_numel, w_k, h_k, w_stride, h_stride, info, src, dst)

extern "C" __global__ void avg_pool2d_f32(size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void avg_pool2d_f64(size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void avg_pool2d_u8(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void avg_pool2d_u32(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void avg_pool2d_f16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void avg_pool2d_bf16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for max_pool2d kernels
// =============================================================================
// Signature: (src_numel, w_k, h_k, w_stride, h_stride, info, src, dst)

extern "C" __global__ void max_pool2d_f32(size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void max_pool2d_f64(size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void max_pool2d_u8(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void max_pool2d_u32(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void max_pool2d_f16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void max_pool2d_bf16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Forward declarations for upsample_nearest2d kernels
// =============================================================================
// Signature: (w_out, h_out, w_scale, h_scale, info, src, dst)

extern "C" __global__ void upsample_nearest2d_f32(size_t, size_t, double, double, const size_t*, const float*, float*);
extern "C" __global__ void upsample_nearest2d_f64(size_t, size_t, double, double, const size_t*, const double*, double*);
extern "C" __global__ void upsample_nearest2d_u8(size_t, size_t, double, double, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void upsample_nearest2d_u32(size_t, size_t, double, double, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void upsample_nearest2d_f16(size_t, size_t, double, double, const size_t*, const __half*, __half*);
extern "C" __global__ void upsample_nearest2d_bf16(size_t, size_t, double, double, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// =============================================================================
// Dispatcher for conv1d operations
// =============================================================================

/// Dispatcher for 1D convolution.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of output elements (for grid sizing)
/// @param src_numel Total number of source elements
/// @param l_out Output length
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param kernel Convolution kernel
/// @param dst Destination tensor
extern "C" void run_conv1d(
    int32_t dtype,
    size_t dst_numel,
    size_t src_numel,
    size_t l_out,
    size_t stride,
    size_t padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    const void* kernel,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            conv1d_f32<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                       (const float*)src, (const float*)kernel, (float*)dst);
            break;
        case CONV_F64:
            conv1d_f64<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                       (const double*)src, (const double*)kernel, (double*)dst);
            break;
        case CONV_F16:
            conv1d_f16<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                       (const __half*)src, (const __half*)kernel, (__half*)dst);
            break;
        case CONV_BF16:
            conv1d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                        (const __nv_bfloat16*)src, (const __nv_bfloat16*)kernel, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            conv1d_u8<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                      (const uint8_t*)src, (const uint8_t*)kernel, (uint8_t*)dst);
            break;
        case CONV_U32:
            conv1d_u32<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, dilation, info,
                       (const uint32_t*)src, (const uint32_t*)kernel, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for conv2d operations
// =============================================================================

/// Dispatcher for 2D convolution.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of output elements (for grid sizing)
/// @param src_numel Total number of source elements
/// @param w_out Output width
/// @param h_out Output height
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param kernel Convolution kernel
/// @param dst Destination tensor
extern "C" void run_conv2d(
    int32_t dtype,
    size_t dst_numel,
    size_t src_numel,
    size_t w_out,
    size_t h_out,
    size_t stride,
    size_t padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    const void* kernel,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            conv2d_f32<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                       (const float*)src, (const float*)kernel, (float*)dst);
            break;
        case CONV_F64:
            conv2d_f64<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                       (const double*)src, (const double*)kernel, (double*)dst);
            break;
        case CONV_F16:
            conv2d_f16<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                       (const __half*)src, (const __half*)kernel, (__half*)dst);
            break;
        case CONV_BF16:
            conv2d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                        (const __nv_bfloat16*)src, (const __nv_bfloat16*)kernel, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            conv2d_u8<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                      (const uint8_t*)src, (const uint8_t*)kernel, (uint8_t*)dst);
            break;
        case CONV_U32:
            conv2d_u32<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, dilation, info,
                       (const uint32_t*)src, (const uint32_t*)kernel, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for conv_transpose1d operations
// =============================================================================

/// Dispatcher for 1D transposed convolution.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of output elements (for grid sizing)
/// @param src_numel Total number of source elements
/// @param l_out Output length
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param out_padding Output padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param kernel Convolution kernel
/// @param dst Destination tensor
extern "C" void run_conv_transpose1d(
    int32_t dtype,
    size_t dst_numel,
    size_t src_numel,
    size_t l_out,
    size_t stride,
    size_t padding,
    size_t out_padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    const void* kernel,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            conv_transpose1d_f32<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                 (const float*)src, (const float*)kernel, (float*)dst);
            break;
        case CONV_F64:
            conv_transpose1d_f64<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                 (const double*)src, (const double*)kernel, (double*)dst);
            break;
        case CONV_F16:
            conv_transpose1d_f16<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                 (const __half*)src, (const __half*)kernel, (__half*)dst);
            break;
        case CONV_BF16:
            conv_transpose1d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                  (const __nv_bfloat16*)src, (const __nv_bfloat16*)kernel, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            conv_transpose1d_u8<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                (const uint8_t*)src, (const uint8_t*)kernel, (uint8_t*)dst);
            break;
        case CONV_U32:
            conv_transpose1d_u32<<<grid, BLOCK_SIZE>>>(src_numel, l_out, stride, padding, out_padding, dilation, info,
                                 (const uint32_t*)src, (const uint32_t*)kernel, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for conv_transpose2d operations
// =============================================================================

/// Dispatcher for 2D transposed convolution.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of output elements (for grid sizing)
/// @param src_numel Total number of source elements
/// @param w_out Output width
/// @param h_out Output height
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param out_padding Output padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param kernel Convolution kernel
/// @param dst Destination tensor
extern "C" void run_conv_transpose2d(
    int32_t dtype,
    size_t dst_numel,
    size_t src_numel,
    size_t w_out,
    size_t h_out,
    size_t stride,
    size_t padding,
    size_t out_padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    const void* kernel,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            conv_transpose2d_f32<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                 (const float*)src, (const float*)kernel, (float*)dst);
            break;
        case CONV_F64:
            conv_transpose2d_f64<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                 (const double*)src, (const double*)kernel, (double*)dst);
            break;
        case CONV_F16:
            conv_transpose2d_f16<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                 (const __half*)src, (const __half*)kernel, (__half*)dst);
            break;
        case CONV_BF16:
            conv_transpose2d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                  (const __nv_bfloat16*)src, (const __nv_bfloat16*)kernel, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            conv_transpose2d_u8<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                (const uint8_t*)src, (const uint8_t*)kernel, (uint8_t*)dst);
            break;
        case CONV_U32:
            conv_transpose2d_u32<<<grid, BLOCK_SIZE>>>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info,
                                 (const uint32_t*)src, (const uint32_t*)kernel, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for im2col operations (2D)
// =============================================================================

/// Dispatcher for 2D im2col transformation.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of destination elements
/// @param h_out Output height
/// @param w_out Output width
/// @param h_k Kernel height
/// @param w_k Kernel width
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_im2col(
    int32_t dtype,
    size_t dst_numel,
    size_t h_out,
    size_t w_out,
    size_t h_k,
    size_t w_k,
    size_t stride,
    size_t padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            im2col_f32<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                       (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            im2col_f64<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                       (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            im2col_f16<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                       (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            im2col_bf16<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                        (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            im2col_u8<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                      (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            im2col_u32<<<grid, BLOCK_SIZE>>>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info,
                       (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for im2col1d operations
// =============================================================================

/// Dispatcher for 1D im2col transformation.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_numel Total number of destination elements
/// @param l_out Output length
/// @param l_k Kernel length
/// @param stride Convolution stride
/// @param padding Convolution padding
/// @param dilation Convolution dilation
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_im2col1d(
    int32_t dtype,
    size_t dst_numel,
    size_t l_out,
    size_t l_k,
    size_t stride,
    size_t padding,
    size_t dilation,
    const size_t* info,
    const void* src,
    void* dst
) {
    int grid = grid_size(dst_numel);
    switch (dtype) {
        case CONV_F32:
            im2col1d_f32<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                         (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            im2col1d_f64<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                         (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            im2col1d_f16<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                         (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            im2col1d_bf16<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                          (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            im2col1d_u8<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                        (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            im2col1d_u32<<<grid, BLOCK_SIZE>>>(dst_numel, l_out, l_k, stride, padding, dilation, info,
                         (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for col2im1d operations
// =============================================================================

/// Dispatcher for 1D col2im transformation.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param dst_el Total number of destination elements
/// @param l_out Output length
/// @param l_in Input length
/// @param c_out Output channels
/// @param k_size Kernel size
/// @param stride Convolution stride
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_col2im1d(
    int32_t dtype,
    size_t dst_el,
    size_t l_out,
    size_t l_in,
    size_t c_out,
    size_t k_size,
    size_t stride,
    const void* src,
    void* dst
) {
    int grid = grid_size(dst_el);
    switch (dtype) {
        case CONV_F32:
            col2im1d_f32<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                         (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            col2im1d_f64<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                         (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            col2im1d_f16<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                         (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            col2im1d_bf16<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                          (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            col2im1d_u8<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                        (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            col2im1d_u32<<<grid, BLOCK_SIZE>>>(dst_el, l_out, l_in, c_out, k_size, stride,
                         (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for avg_pool2d operations
// =============================================================================

/// Dispatcher for 2D average pooling.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param src_numel Total number of source elements
/// @param w_k Kernel width
/// @param h_k Kernel height
/// @param w_stride Width stride
/// @param h_stride Height stride
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_avg_pool2d(
    int32_t dtype,
    size_t src_numel,
    size_t w_k,
    size_t h_k,
    size_t w_stride,
    size_t h_stride,
    const size_t* info,
    const void* src,
    void* dst
) {
    int grid = grid_size(src_numel);
    switch (dtype) {
        case CONV_F32:
            avg_pool2d_f32<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            avg_pool2d_f64<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            avg_pool2d_f16<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            avg_pool2d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                            (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            avg_pool2d_u8<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                          (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            avg_pool2d_u32<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for max_pool2d operations
// =============================================================================

/// Dispatcher for 2D max pooling.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param src_numel Total number of source elements
/// @param w_k Kernel width
/// @param h_k Kernel height
/// @param w_stride Width stride
/// @param h_stride Height stride
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_max_pool2d(
    int32_t dtype,
    size_t src_numel,
    size_t w_k,
    size_t h_k,
    size_t w_stride,
    size_t h_stride,
    const size_t* info,
    const void* src,
    void* dst
) {
    int grid = grid_size(src_numel);
    switch (dtype) {
        case CONV_F32:
            max_pool2d_f32<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            max_pool2d_f64<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            max_pool2d_f16<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            max_pool2d_bf16<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                            (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            max_pool2d_u8<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                          (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            max_pool2d_u32<<<grid, BLOCK_SIZE>>>(src_numel, w_k, h_k, w_stride, h_stride, info,
                           (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}

// =============================================================================
// Dispatcher for upsample_nearest2d operations
// =============================================================================

/// Dispatcher for 2D nearest neighbor upsampling.
/// @param dtype Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u8, 5=u32)
/// @param w_out Output width
/// @param h_out Output height
/// @param w_scale Width scale factor
/// @param h_scale Height scale factor
/// @param info Pointer to dims and strides info
/// @param src Source tensor
/// @param dst Destination tensor
extern "C" void run_upsample_nearest2d(
    int32_t dtype,
    size_t w_out,
    size_t h_out,
    double w_scale,
    double h_scale,
    const size_t* info,
    const void* src,
    void* dst
) {
    // Use output size to calculate grid
    size_t numel = w_out * h_out;
    int grid = grid_size(numel);
    switch (dtype) {
        case CONV_F32:
            upsample_nearest2d_f32<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                   (const float*)src, (float*)dst);
            break;
        case CONV_F64:
            upsample_nearest2d_f64<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                   (const double*)src, (double*)dst);
            break;
        case CONV_F16:
            upsample_nearest2d_f16<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                   (const __half*)src, (__half*)dst);
            break;
        case CONV_BF16:
            upsample_nearest2d_bf16<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                    (const __nv_bfloat16*)src, (__nv_bfloat16*)dst);
            break;
        case CONV_U8:
            upsample_nearest2d_u8<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                  (const uint8_t*)src, (uint8_t*)dst);
            break;
        case CONV_U32:
            upsample_nearest2d_u32<<<grid, BLOCK_SIZE>>>(w_out, h_out, w_scale, h_scale, info,
                                   (const uint32_t*)src, (uint32_t*)dst);
            break;
    }
}
