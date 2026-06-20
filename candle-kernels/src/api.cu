// =============================================================================
// CANDLE-KERNELS UNIFIED API
// =============================================================================
// This file provides extern "C" wrapper functions for all candle-kernels.
// Rust calls these wrappers via FFI, and the wrappers launch the CUDA kernels.
//
// Organization:
//   1. Simple kernels (affine, binary, unary, etc.) - direct wrappers
//   2. Quantized kernels - dispatch by quant type
//   3. Flash-attention kernels - already have their own api.cu files
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <cstddef>

// =============================================================================
// KERNEL LAUNCH CONFIGURATION
// =============================================================================
// Default block size for element-wise operations
constexpr int BLOCK_SIZE = 256;

// Calculate grid size for a given number of elements
inline int grid_size(size_t numel, int block_size = BLOCK_SIZE) {
    return (numel + block_size - 1) / block_size;
}

// =============================================================================
// FORWARD DECLARATIONS OF __global__ KERNELS
// =============================================================================
// These are defined in the individual .cu files with extern "C" __global__

// --- Affine kernels ---
extern "C" __global__ void affine_f32(const size_t, const size_t, const size_t*, const float*, float*, const float, const float);
extern "C" __global__ void affine_f64(const size_t, const size_t, const size_t*, const double*, double*, const double, const double);
extern "C" __global__ void affine_f16(const size_t, const size_t, const size_t*, const __half*, __half*, const __half, const __half);
extern "C" __global__ void affine_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16, const __nv_bfloat16);
extern "C" __global__ void affine_u8(const size_t, const size_t, const size_t*, const uint8_t*, uint8_t*, const uint8_t, const uint8_t);
extern "C" __global__ void affine_u32(const size_t, const size_t, const size_t*, const uint32_t*, uint32_t*, const uint32_t, const uint32_t);
extern "C" __global__ void affine_i16(const size_t, const size_t, const size_t*, const int16_t*, int16_t*, const int16_t, const int16_t);
extern "C" __global__ void affine_i32(const size_t, const size_t, const size_t*, const int32_t*, int32_t*, const int32_t, const int32_t);
extern "C" __global__ void affine_i64(const size_t, const size_t, const size_t*, const int64_t*, int64_t*, const int64_t, const int64_t);

// --- Fill kernels ---
extern "C" __global__ void fill_u8(uint8_t*, uint8_t, const size_t);
extern "C" __global__ void fill_u32(uint32_t*, uint32_t, const size_t);
extern "C" __global__ void fill_i64(int64_t*, int64_t, const size_t);
extern "C" __global__ void fill_f32(float*, float, const size_t);
extern "C" __global__ void fill_f64(double*, double, const size_t);
extern "C" __global__ void fill_f16(__half*, __half, const size_t);
extern "C" __global__ void fill_bf16(__nv_bfloat16*, __nv_bfloat16, const size_t);

// --- Binary kernels (selected) ---
extern "C" __global__ void badd_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void badd_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void badd_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void badd_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bmul_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void bmul_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void bmul_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void bmul_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bdiv_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void bdiv_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void bdiv_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void bdiv_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bsub_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void bsub_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void bsub_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void bsub_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bminimum_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void bmaximum_f32(const size_t, const size_t, const size_t*, const float*, const float*, float*);

// --- Unary kernels (selected) ---
extern "C" __global__ void ucopy_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ucopy_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ucopy_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ucopy_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ucopy_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void uneg_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void uexp_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ulog_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usin_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ucos_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void utanh_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usqrt_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ugelu_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void urelu_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usilu_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usigmoid_f32(const size_t, const size_t, const size_t*, const float*, float*);

// --- Reduce kernels ---
extern "C" __global__ void fast_sum_f32(const size_t, const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_max_f32(const size_t, const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_min_f32(const size_t, const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_argmax_f32(const size_t, const size_t, const size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void fast_argmin_f32(const size_t, const size_t, const size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void softmax_f32(const float*, float*, const int, const int);

// --- Indexing kernels ---
extern "C" __global__ void is_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f32(const size_t, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f32(const uint32_t*, const size_t, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f32(const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);

// --- Where (ternary) kernels ---
// where_i64_* (condition: int64_t)
extern "C" __global__ void where_i64_f32(const size_t, const size_t, const size_t*, const int64_t*, const float*, const float*, float*);
extern "C" __global__ void where_i64_f64(const size_t, const size_t, const size_t*, const int64_t*, const double*, const double*, double*);
extern "C" __global__ void where_i64_u8(const size_t, const size_t, const size_t*, const int64_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void where_i64_u32(const size_t, const size_t, const size_t*, const int64_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void where_i64_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void where_i64_f16(const size_t, const size_t, const size_t*, const int64_t*, const __half*, const __half*, __half*);
extern "C" __global__ void where_i64_bf16(const size_t, const size_t, const size_t*, const int64_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
// where_u32_* (condition: uint32_t)
extern "C" __global__ void where_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, const float*, const float*, float*);
extern "C" __global__ void where_u32_f64(const size_t, const size_t, const size_t*, const uint32_t*, const double*, const double*, double*);
extern "C" __global__ void where_u32_u8(const size_t, const size_t, const size_t*, const uint32_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void where_u32_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void where_u32_i64(const size_t, const size_t, const size_t*, const uint32_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void where_u32_f16(const size_t, const size_t, const size_t*, const uint32_t*, const __half*, const __half*, __half*);
extern "C" __global__ void where_u32_bf16(const size_t, const size_t, const size_t*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
// where_u8_* (condition: uint8_t)
extern "C" __global__ void where_u8_f32(const size_t, const size_t, const size_t*, const uint8_t*, const float*, const float*, float*);
extern "C" __global__ void where_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, const double*, const double*, double*);
extern "C" __global__ void where_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void where_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void where_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void where_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, const __half*, const __half*, __half*);
extern "C" __global__ void where_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

// --- Conv kernels ---
// im2col (2D): (dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst)
extern "C" __global__ void im2col_f32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void im2col_f64(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void im2col_f16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void im2col_bf16(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void im2col_u8(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void im2col_u32(size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);

// im2col1d: (dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst)
extern "C" __global__ void im2col1d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void im2col1d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void im2col1d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void im2col1d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void im2col1d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void im2col1d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);

// col2im1d: (dst_el, l_out, l_in, c_out, k_size, stride, src, dst)
extern "C" __global__ void col2im1d_f32(size_t, size_t, size_t, size_t, size_t, size_t, const float*, float*);
extern "C" __global__ void col2im1d_f64(size_t, size_t, size_t, size_t, size_t, size_t, const double*, double*);
extern "C" __global__ void col2im1d_f16(size_t, size_t, size_t, size_t, size_t, size_t, const __half*, __half*);
extern "C" __global__ void col2im1d_bf16(size_t, size_t, size_t, size_t, size_t, size_t, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void col2im1d_u8(size_t, size_t, size_t, size_t, size_t, size_t, const uint8_t*, uint8_t*);
extern "C" __global__ void col2im1d_u32(size_t, size_t, size_t, size_t, size_t, size_t, const uint32_t*, uint32_t*);

// upsample_nearest2d: (w_out, h_out, w_scale, h_scale, info, src, dst)
extern "C" __global__ void upsample_nearest2d_f32(size_t, size_t, double, double, const size_t*, const float*, float*);
extern "C" __global__ void upsample_nearest2d_f64(size_t, size_t, double, double, const size_t*, const double*, double*);
extern "C" __global__ void upsample_nearest2d_f16(size_t, size_t, double, double, const size_t*, const __half*, __half*);
extern "C" __global__ void upsample_nearest2d_bf16(size_t, size_t, double, double, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void upsample_nearest2d_u8(size_t, size_t, double, double, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void upsample_nearest2d_u32(size_t, size_t, double, double, const size_t*, const uint32_t*, uint32_t*);

// avg_pool2d: (src_numel, w_k, h_k, w_stride, h_stride, info, src, dst)
extern "C" __global__ void avg_pool2d_f32(size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void avg_pool2d_f64(size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void avg_pool2d_f16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void avg_pool2d_bf16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void avg_pool2d_u8(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void avg_pool2d_u32(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);

// max_pool2d: (src_numel, w_k, h_k, w_stride, h_stride, info, src, dst)
extern "C" __global__ void max_pool2d_f32(size_t, size_t, size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void max_pool2d_f64(size_t, size_t, size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void max_pool2d_f16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void max_pool2d_bf16(size_t, size_t, size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void max_pool2d_u8(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void max_pool2d_u32(size_t, size_t, size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);

// --- Fast reduce kernels ---
// fast_sum: (src_numel, el_to_sum_per_block, num_dims, info, src, dst)
extern "C" __global__ void fast_sum_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_sum_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void fast_sum_f16(size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void fast_sum_bf16(size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fast_sum_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void fast_sum_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void fast_sum_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void fast_sum_f8_e4m3(size_t, size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// fast_min: (src_numel, el_to_sum_per_block, num_dims, info, src, dst)
extern "C" __global__ void fast_min_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_min_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void fast_min_f16(size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void fast_min_bf16(size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fast_min_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void fast_min_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void fast_min_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void fast_min_f8_e4m3(size_t, size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// fast_max: (src_numel, el_to_sum_per_block, num_dims, info, src, dst)
extern "C" __global__ void fast_max_f32(size_t, size_t, size_t, const size_t*, const float*, float*);
extern "C" __global__ void fast_max_f64(size_t, size_t, size_t, const size_t*, const double*, double*);
extern "C" __global__ void fast_max_f16(size_t, size_t, size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void fast_max_bf16(size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fast_max_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void fast_max_i64(size_t, size_t, size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void fast_max_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void fast_max_f8_e4m3(size_t, size_t, size_t, const size_t*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// fast_argmin/argmax: (src_numel, el_to_sum_per_block, num_dims, info, src, dst)
extern "C" __global__ void fast_argmin_f32(size_t, size_t, size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void fast_argmin_f64(size_t, size_t, size_t, const size_t*, const double*, uint32_t*);
extern "C" __global__ void fast_argmin_f16(size_t, size_t, size_t, const size_t*, const __half*, uint32_t*);
extern "C" __global__ void fast_argmin_bf16(size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, uint32_t*);
extern "C" __global__ void fast_argmin_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void fast_argmin_i64(size_t, size_t, size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" __global__ void fast_argmin_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" __global__ void fast_argmin_f8_e4m3(size_t, size_t, size_t, const size_t*, const __nv_fp8_e4m3*, uint32_t*);

extern "C" __global__ void fast_argmax_f32(size_t, size_t, size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void fast_argmax_f64(size_t, size_t, size_t, const size_t*, const double*, uint32_t*);
extern "C" __global__ void fast_argmax_f16(size_t, size_t, size_t, const size_t*, const __half*, uint32_t*);
extern "C" __global__ void fast_argmax_bf16(size_t, size_t, size_t, const size_t*, const __nv_bfloat16*, uint32_t*);
extern "C" __global__ void fast_argmax_u32(size_t, size_t, size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void fast_argmax_i64(size_t, size_t, size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" __global__ void fast_argmax_u8(size_t, size_t, size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" __global__ void fast_argmax_f8_e4m3(size_t, size_t, size_t, const size_t*, const __nv_fp8_e4m3*, uint32_t*);

// --- Sort kernels ---
extern "C" __global__ void asort_asc_f32(const float*, uint32_t*, const size_t, const size_t, const size_t);
extern "C" __global__ void asort_desc_f32(const float*, uint32_t*, const size_t, const size_t, const size_t);

// --- Cast kernels ---
extern "C" __global__ void cast_f32_f16(const size_t, const size_t, const size_t*, const float*, __half*);
extern "C" __global__ void cast_f16_f32(const size_t, const size_t, const size_t*, const __half*, float*);
extern "C" __global__ void cast_f32_bf16(const size_t, const size_t, const size_t*, const float*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_f32(const size_t, const size_t, const size_t*, const __nv_bfloat16*, float*);

// --- Quantized matmul kernels ---
extern "C" __global__ void quantize_q8_1(const float*, void*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q4_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q4_1_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_1_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q8_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q2_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q3_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q4_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_k(const void*, const float*, float*, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q6_k(const void*, const float*, float*, const int, const int);

// --- Multinomial kernel ---
extern "C" __global__ void optimized_multinomial_f32(const float*, uint32_t*, float*, const uint32_t, const uint32_t, const float, const float, const uint64_t);

// =============================================================================
// WRAPPER FUNCTIONS
// =============================================================================

extern "C" {

// --- Affine wrappers ---
void run_affine_f32(
    const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add
) {
    affine_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_f64(
    const double* inp, double* out,
    size_t numel, size_t num_dims, const size_t* info,
    double mul, double add
) {
    affine_f64<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_f16(
    const void* inp, void* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add
) {
    __half h_mul = __float2half(mul);
    __half h_add = __float2half(add);
    affine_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, (const __half*)inp, (__half*)out, h_mul, h_add);
}

void run_affine_bf16(
    const void* inp, void* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add
) {
    __nv_bfloat16 h_mul = __float2bfloat16(mul);
    __nv_bfloat16 h_add = __float2bfloat16(add);
    affine_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out, h_mul, h_add);
}

void run_affine_u8(
    const uint8_t* inp, uint8_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    uint8_t mul, uint8_t add
) {
    affine_u8<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_u32(
    const uint32_t* inp, uint32_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    uint32_t mul, uint32_t add
) {
    affine_u32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i16(
    const int16_t* inp, int16_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int16_t mul, int16_t add
) {
    affine_i16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i32(
    const int32_t* inp, int32_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int32_t mul, int32_t add
) {
    affine_i32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i64(
    const int64_t* inp, int64_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int64_t mul, int64_t add
) {
    affine_i64<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out, mul, add);
}

// --- Fill wrappers ---
void run_fill_f32(float* buf, float value, size_t numel) {
    fill_f32<<<grid_size(numel), BLOCK_SIZE>>>(buf, value, numel);
}

void run_fill_f64(double* buf, double value, size_t numel) {
    fill_f64<<<grid_size(numel), BLOCK_SIZE>>>(buf, value, numel);
}

void run_fill_f16(void* buf, float value, size_t numel) {
    __half h_val = __float2half(value);
    fill_f16<<<grid_size(numel), BLOCK_SIZE>>>((__half*)buf, h_val, numel);
}

void run_fill_bf16(void* buf, float value, size_t numel) {
    __nv_bfloat16 h_val = __float2bfloat16(value);
    fill_bf16<<<grid_size(numel), BLOCK_SIZE>>>((__nv_bfloat16*)buf, h_val, numel);
}

void run_fill_u8(uint8_t* buf, uint8_t value, size_t numel) {
    fill_u8<<<grid_size(numel), BLOCK_SIZE>>>(buf, value, numel);
}

void run_fill_u32(uint32_t* buf, uint32_t value, size_t numel) {
    fill_u32<<<grid_size(numel), BLOCK_SIZE>>>(buf, value, numel);
}

void run_fill_i64(int64_t* buf, int64_t value, size_t numel) {
    fill_i64<<<grid_size(numel), BLOCK_SIZE>>>(buf, value, numel);
}

// --- Binary operation wrappers ---
void run_binary_f32(
    const float* lhs, const float* rhs, float* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op  // 0=add, 1=mul, 2=div, 3=sub, 4=min, 5=max
) {
    switch (op) {
        case 0: badd_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 1: bmul_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 2: bdiv_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 3: bsub_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 4: bminimum_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 5: bmaximum_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
    }
}

void run_binary_f16(
    const void* lhs, const void* rhs, void* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op
) {
    switch (op) {
        case 0: badd_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 1: bmul_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 2: bdiv_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 3: bsub_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
    }
}

void run_binary_bf16(
    const void* lhs, const void* rhs, void* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op
) {
    switch (op) {
        case 0: badd_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 1: bmul_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 2: bdiv_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 3: bsub_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
    }
}

// --- Unary operation wrappers ---
void run_unary_f32(
    const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    int op  // 0=copy, 1=neg, 2=exp, 3=log, 4=sin, 5=cos, 6=tanh, 7=sqrt, 8=gelu, 9=relu, 10=silu, 11=sigmoid
) {
    switch (op) {
        case 0: ucopy_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 1: uneg_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 2: uexp_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 3: ulog_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 4: usin_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 5: ucos_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 6: utanh_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 7: usqrt_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 8: ugelu_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 9: urelu_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 10: usilu_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
        case 11: usigmoid_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, out); break;
    }
}

// --- Reduce wrappers ---
void run_reduce_sum_f32(
    const float* src, float* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_sum_f32<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_reduce_max_f32(
    const float* src, float* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_max_f32<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_reduce_argmax_f32(
    const float* src, uint32_t* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_argmax_f32<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_softmax_f32(const float* src, float* dst, int ncols, int nrows) {
    softmax_f32<<<nrows, BLOCK_SIZE>>>(src, dst, ncols, nrows);
}

// --- Where (ternary) wrappers ---
void run_where_u8_f32(
    const uint8_t* ids, const float* t, const float* f, float* out,
    size_t numel, size_t num_dims, const size_t* info
) {
    where_u8_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, ids, t, f, out);
}

// --- Index select wrapper ---
void run_index_select_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size
) {
    is_u32_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size);
}

// --- Gather wrapper ---
void run_gather_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t numel, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size
) {
    gather_u32_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size);
}

// --- Index add wrapper ---
void run_index_add_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t ids_dim_size, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size
) {
    size_t numel = left_size * right_size;
    ia_u32_f32<<<grid_size(numel), BLOCK_SIZE>>>(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size);
}

// --- Scatter add wrapper ---
void run_scatter_add_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size
) {
    size_t numel = left_size * right_size;
    sa_u32_f32<<<grid_size(numel), BLOCK_SIZE>>>(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);
}

// --- Sort wrappers ---
void run_argsort_asc_f32(const float* src, uint32_t* dst, size_t nrows, size_t ncols, size_t ncols_pad) {
    asort_asc_f32<<<nrows, BLOCK_SIZE>>>(src, dst, nrows, ncols, ncols_pad);
}

void run_argsort_desc_f32(const float* src, uint32_t* dst, size_t nrows, size_t ncols, size_t ncols_pad) {
    asort_desc_f32<<<nrows, BLOCK_SIZE>>>(src, dst, nrows, ncols, ncols_pad);
}

// --- Cast wrappers ---
void run_cast_f32_f16(const float* inp, void* out, size_t numel, size_t num_dims, const size_t* info) {
    cast_f32_f16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, (__half*)out);
}

void run_cast_f16_f32(const void* inp, float* out, size_t numel, size_t num_dims, const size_t* info) {
    cast_f16_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, (const __half*)inp, out);
}

void run_cast_f32_bf16(const float* inp, void* out, size_t numel, size_t num_dims, const size_t* info) {
    cast_f32_bf16<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, inp, (__nv_bfloat16*)out);
}

void run_cast_bf16_f32(const void* inp, float* out, size_t numel, size_t num_dims, const size_t* info) {
    cast_bf16_f32<<<grid_size(numel), BLOCK_SIZE>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, out);
}

// NOTE: run_quantize_q8_1 and run_dequantize_mul_mat_vec are defined in quantized_dispatcher.cu

// --- Multinomial wrapper ---
void run_multinomial_f32(
    const float* logits, uint32_t* output, float* workspace,
    uint32_t vocab_size, uint32_t top_k, float top_p, float temperature, uint64_t seed
) {
    optimized_multinomial_f32<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        logits, output, workspace, vocab_size, top_k, top_p, temperature, seed
    );
}

// --- RoPE wrappers ---
// These need special handling due to different signatures

// =============================================================================
// ADDITIONAL KERNEL DECLARATIONS FOR DISPATCHERS
// =============================================================================

// --- Additional binary arithmetic kernels ---
extern "C" __global__ void badd_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void badd_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void badd_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void bmul_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void bmul_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void bmul_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void bdiv_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void bdiv_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void bdiv_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void bsub_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void bsub_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void bsub_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void bminimum_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void bminimum_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void bminimum_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bminimum_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void bminimum_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void bminimum_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
extern "C" __global__ void bmaximum_f64(const size_t, const size_t, const size_t*, const double*, const double*, double*);
extern "C" __global__ void bmaximum_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void bmaximum_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void bmaximum_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void bmaximum_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void bmaximum_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*);
// f8_e4m3 binary arithmetic kernels
extern "C" __global__ void badd_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void bmul_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void bdiv_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void bsub_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void bminimum_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
extern "C" __global__ void bmaximum_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// --- Binary comparison kernels ---
extern "C" __global__ void eq_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void eq_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void eq_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void eq_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void eq_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void eq_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void eq_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
extern "C" __global__ void ne_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void ne_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void ne_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void ne_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void ne_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void ne_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void ne_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
extern "C" __global__ void lt_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void lt_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void lt_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void lt_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void lt_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void lt_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void lt_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
extern "C" __global__ void le_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void le_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void le_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void le_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void le_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void le_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void le_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
extern "C" __global__ void gt_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void gt_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void gt_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void gt_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void gt_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void gt_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void gt_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
extern "C" __global__ void ge_f32(const size_t, const size_t, const size_t*, const float*, const float*, uint8_t*);
extern "C" __global__ void ge_f64(const size_t, const size_t, const size_t*, const double*, const double*, uint8_t*);
extern "C" __global__ void ge_f16(const size_t, const size_t, const size_t*, const __half*, const __half*, uint8_t*);
extern "C" __global__ void ge_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void ge_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void ge_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void ge_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, uint8_t*);
// f8_e4m3 binary comparison kernels
extern "C" __global__ void eq_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void ne_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void lt_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void le_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void gt_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);
extern "C" __global__ void ge_f8_e4m3(const size_t, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*);

// --- In-place binary arithmetic kernels ---
// Signature: (numel, num_dims, dims_and_strides, lhs_mut, rhs)
extern "C" __global__ void badd_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void bsub_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void bmul_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void bdiv_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void bmin_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void bmax_f32_inplace(const size_t, const size_t, const size_t*, float*, const float*);
extern "C" __global__ void badd_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void bsub_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void bmul_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void bdiv_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void bmin_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void bmax_f64_inplace(const size_t, const size_t, const size_t*, double*, const double*);
extern "C" __global__ void badd_u8_inplace(const size_t, const size_t, const size_t*, uint8_t*, const uint8_t*);
extern "C" __global__ void bsub_u8_inplace(const size_t, const size_t, const size_t*, uint8_t*, const uint8_t*);
extern "C" __global__ void bmul_u8_inplace(const size_t, const size_t, const size_t*, uint8_t*, const uint8_t*);
extern "C" __global__ void bdiv_u8_inplace(const size_t, const size_t, const size_t*, uint8_t*, const uint8_t*);
extern "C" __global__ void badd_u32_inplace(const size_t, const size_t, const size_t*, uint32_t*, const uint32_t*);
extern "C" __global__ void bsub_u32_inplace(const size_t, const size_t, const size_t*, uint32_t*, const uint32_t*);
extern "C" __global__ void bmul_u32_inplace(const size_t, const size_t, const size_t*, uint32_t*, const uint32_t*);
extern "C" __global__ void bdiv_u32_inplace(const size_t, const size_t, const size_t*, uint32_t*, const uint32_t*);
extern "C" __global__ void badd_i64_inplace(const size_t, const size_t, const size_t*, int64_t*, const int64_t*);
extern "C" __global__ void bsub_i64_inplace(const size_t, const size_t, const size_t*, int64_t*, const int64_t*);
extern "C" __global__ void bmul_i64_inplace(const size_t, const size_t, const size_t*, int64_t*, const int64_t*);
extern "C" __global__ void bdiv_i64_inplace(const size_t, const size_t, const size_t*, int64_t*, const int64_t*);
extern "C" __global__ void badd_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void bsub_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void bmul_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void bdiv_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void bmin_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void bmax_f16_inplace(const size_t, const size_t, const size_t*, __half*, const __half*);
extern "C" __global__ void badd_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void bsub_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void bmul_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void bdiv_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void bmin_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void bmax_bf16_inplace(const size_t, const size_t, const size_t*, __nv_bfloat16*, const __nv_bfloat16*);
extern "C" __global__ void badd_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);
extern "C" __global__ void bsub_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);
extern "C" __global__ void bmul_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);
extern "C" __global__ void bdiv_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);
extern "C" __global__ void bmin_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);
extern "C" __global__ void bmax_f8_e4m3_inplace(const size_t, const size_t, const size_t*, __nv_fp8_e4m3*, const __nv_fp8_e4m3*);

// --- Additional unary kernels ---
extern "C" __global__ void ucopy_u8(const size_t, const size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void ucopy_u32(const size_t, const size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void ucopy_i64(const size_t, const size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void uneg_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uneg_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uneg_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void urecip_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void urecip_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void urecip_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void urecip_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void uexp_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uexp_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uexp_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ulog_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ulog_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ulog_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usin_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usin_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usin_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ucos_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ucos_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ucos_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void utanh_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void utanh_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void utanh_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void uerf_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void uerf_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uerf_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uerf_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void uceil_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void uceil_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uceil_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uceil_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ufloor_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ufloor_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ufloor_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ufloor_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void uround_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void uround_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uround_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uround_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void unormcdf_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void unormcdf_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void unormcdf_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void unormcdf_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void uabs_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void uabs_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void uabs_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void uabs_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usqr_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usqr_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usqr_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usqr_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usqrt_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usqrt_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usqrt_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ugelu_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ugelu_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ugelu_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void ugelu_erf_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void ugelu_erf_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void ugelu_erf_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void ugelu_erf_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void urelu_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void urelu_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void urelu_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usilu_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usilu_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usilu_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usign_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void usign_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usign_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usign_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void usigmoid_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void usigmoid_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void usigmoid_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);

// --- Parametric unary kernels ---
extern "C" __global__ void uelu_f32(const size_t, const size_t, const size_t*, float, const float*, float*);
extern "C" __global__ void uelu_f64(const size_t, const size_t, const size_t*, double, const double*, double*);
extern "C" __global__ void uelu_f16(const size_t, const size_t, const size_t*, __half, const __half*, __half*);
extern "C" __global__ void uelu_bf16(const size_t, const size_t, const size_t*, __nv_bfloat16, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void upowf_f32(const size_t, const size_t, const size_t*, float, const float*, float*);
extern "C" __global__ void upowf_f64(const size_t, const size_t, const size_t*, double, const double*, double*);
extern "C" __global__ void upowf_f16(const size_t, const size_t, const size_t*, __half, const __half*, __half*);
extern "C" __global__ void upowf_bf16(const size_t, const size_t, const size_t*, __nv_bfloat16, const __nv_bfloat16*, __nv_bfloat16*);

// --- Cast kernels (additional) ---
extern "C" __global__ void cast_f32_f32(const size_t, const size_t, const size_t*, const float*, float*);
extern "C" __global__ void cast_f64_f64(const size_t, const size_t, const size_t*, const double*, double*);
extern "C" __global__ void cast_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void cast_u32_u32(const size_t, const size_t, const size_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void cast_i64_i64(const size_t, const size_t, const size_t*, const int64_t*, int64_t*);
extern "C" __global__ void cast_f64_f32(const size_t, const size_t, const size_t*, const double*, float*);
extern "C" __global__ void cast_f32_f64(const size_t, const size_t, const size_t*, const float*, double*);
extern "C" __global__ void cast_f32_u8(const size_t, const size_t, const size_t*, const float*, uint8_t*);
extern "C" __global__ void cast_u8_f32(const size_t, const size_t, const size_t*, const uint8_t*, float*);
extern "C" __global__ void cast_f32_u32(const size_t, const size_t, const size_t*, const float*, uint32_t*);
extern "C" __global__ void cast_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, float*);
extern "C" __global__ void cast_f32_i64(const size_t, const size_t, const size_t*, const float*, int64_t*);
extern "C" __global__ void cast_i64_f32(const size_t, const size_t, const size_t*, const int64_t*, float*);
extern "C" __global__ void cast_f64_u8(const size_t, const size_t, const size_t*, const double*, uint8_t*);
extern "C" __global__ void cast_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, double*);
extern "C" __global__ void cast_f64_u32(const size_t, const size_t, const size_t*, const double*, uint32_t*);
extern "C" __global__ void cast_u32_f64(const size_t, const size_t, const size_t*, const uint32_t*, double*);
extern "C" __global__ void cast_f64_i64(const size_t, const size_t, const size_t*, const double*, int64_t*);
extern "C" __global__ void cast_i64_f64(const size_t, const size_t, const size_t*, const int64_t*, double*);
extern "C" __global__ void cast_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, uint32_t*);
extern "C" __global__ void cast_u32_u8(const size_t, const size_t, const size_t*, const uint32_t*, uint8_t*);
extern "C" __global__ void cast_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, int64_t*);
extern "C" __global__ void cast_i64_u8(const size_t, const size_t, const size_t*, const int64_t*, uint8_t*);
extern "C" __global__ void cast_u32_i64(const size_t, const size_t, const size_t*, const uint32_t*, int64_t*);
extern "C" __global__ void cast_i64_u32(const size_t, const size_t, const size_t*, const int64_t*, uint32_t*);
extern "C" __global__ void cast_f16_f16(const size_t, const size_t, const size_t*, const __half*, __half*);
extern "C" __global__ void cast_bf16_bf16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void cast_f64_f16(const size_t, const size_t, const size_t*, const double*, __half*);
extern "C" __global__ void cast_f16_f64(const size_t, const size_t, const size_t*, const __half*, double*);
extern "C" __global__ void cast_f64_bf16(const size_t, const size_t, const size_t*, const double*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_f64(const size_t, const size_t, const size_t*, const __nv_bfloat16*, double*);
extern "C" __global__ void cast_f16_bf16(const size_t, const size_t, const size_t*, const __half*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_f16(const size_t, const size_t, const size_t*, const __nv_bfloat16*, __half*);
extern "C" __global__ void cast_f16_u8(const size_t, const size_t, const size_t*, const __half*, uint8_t*);
extern "C" __global__ void cast_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, __half*);
extern "C" __global__ void cast_f16_u32(const size_t, const size_t, const size_t*, const __half*, uint32_t*);
extern "C" __global__ void cast_u32_f16(const size_t, const size_t, const size_t*, const uint32_t*, __half*);
extern "C" __global__ void cast_bf16_u8(const size_t, const size_t, const size_t*, const __nv_bfloat16*, uint8_t*);
extern "C" __global__ void cast_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, __nv_bfloat16*);
extern "C" __global__ void cast_bf16_u32(const size_t, const size_t, const size_t*, const __nv_bfloat16*, uint32_t*);
extern "C" __global__ void cast_u32_bf16(const size_t, const size_t, const size_t*, const uint32_t*, __nv_bfloat16*);

// --- Additional indexing kernels ---
extern "C" __global__ void is_i64_f32(const size_t, const size_t, const size_t*, const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_f64(const size_t, const size_t, const size_t*, const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_f16(const size_t, const size_t, const size_t*, const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_bf16(const size_t, const size_t, const size_t*, const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_u8(const size_t, const size_t, const size_t*, const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_u32(const size_t, const size_t, const size_t*, const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_i64_i64(const size_t, const size_t, const size_t*, const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f32(const size_t, const size_t, const size_t*, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f64(const size_t, const size_t, const size_t*, const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_f16(const size_t, const size_t, const size_t*, const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_bf16(const size_t, const size_t, const size_t*, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_u8(const size_t, const size_t, const size_t*, const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_u32(const size_t, const size_t, const size_t*, const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u32_i64(const size_t, const size_t, const size_t*, const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f32(const size_t, const size_t, const size_t*, const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void is_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void gather_i64_f32(const size_t, const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_f64(const size_t, const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_f16(const size_t, const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_bf16(const size_t, const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_u8(const size_t, const int64_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_u32(const size_t, const int64_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_i64_i64(const size_t, const int64_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f32(const size_t, const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f64(const size_t, const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_f16(const size_t, const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_bf16(const size_t, const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_u8(const size_t, const uint32_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_u32(const size_t, const uint32_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u32_i64(const size_t, const uint32_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f32(const size_t, const uint8_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f64(const size_t, const uint8_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_f16(const size_t, const uint8_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_bf16(const size_t, const uint8_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_u8(const size_t, const uint8_t*, const uint8_t*, uint8_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_u32(const size_t, const uint8_t*, const uint32_t*, uint32_t*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void gather_u8_i64(const size_t, const uint8_t*, const int64_t*, int64_t*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void ia_i64_f32(const int64_t*, const size_t, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_f64(const int64_t*, const size_t, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_f16(const int64_t*, const size_t, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_i64_bf16(const int64_t*, const size_t, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f64(const uint32_t*, const size_t, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_f16(const uint32_t*, const size_t, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void ia_u32_bf16(const uint32_t*, const size_t, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void sa_i64_f32(const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_f64(const int64_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_f16(const int64_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_i64_bf16(const int64_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f64(const uint32_t*, const double*, double*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_f16(const uint32_t*, const __half*, __half*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void sa_u32_bf16(const uint32_t*, const __nv_bfloat16*, __nv_bfloat16*, const size_t, const size_t, const size_t, const size_t);

extern "C" __global__ void s_i64_f32(const int64_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);
extern "C" __global__ void s_u32_f32(const uint32_t*, const float*, float*, const size_t, const size_t, const size_t, const size_t);

// --- Additional where kernels ---
extern "C" __global__ void where_u8_f64(const size_t, const size_t, const size_t*, const uint8_t*, const double*, const double*, double*);
extern "C" __global__ void where_u8_f16(const size_t, const size_t, const size_t*, const uint8_t*, const __half*, const __half*, __half*);
extern "C" __global__ void where_u8_bf16(const size_t, const size_t, const size_t*, const uint8_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void where_u8_u8(const size_t, const size_t, const size_t*, const uint8_t*, const uint8_t*, const uint8_t*, uint8_t*);
extern "C" __global__ void where_u8_u32(const size_t, const size_t, const size_t*, const uint8_t*, const uint32_t*, const uint32_t*, uint32_t*);
extern "C" __global__ void where_u8_i64(const size_t, const size_t, const size_t*, const uint8_t*, const int64_t*, const int64_t*, int64_t*);

// --- Fused SiLU-Mul kernel forward declarations ---
extern "C" __global__ void fused_silu_mul_bf16(const unsigned int, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fused_silu_mul_f16(const unsigned int, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void fused_silu_mul_f32(const unsigned int, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void fused_silu_mul_f8_e4m3(const unsigned int, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);
// Vectorized variants
extern "C" __global__ void fused_silu_mul_bf16_vec2(const unsigned int, const size_t, const size_t*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
extern "C" __global__ void fused_silu_mul_f16_vec2(const unsigned int, const size_t, const size_t*, const __half*, const __half*, __half*);
extern "C" __global__ void fused_silu_mul_f32_vec4(const unsigned int, const size_t, const size_t*, const float*, const float*, float*);
extern "C" __global__ void fused_silu_mul_f8_e4m3_vec4(const unsigned int, const size_t, const size_t*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, __nv_fp8_e4m3*);

// --- Fused MoE gather/scatter kernel forward declarations ---
extern "C" __global__ void moe_gather_bf16(__nv_bfloat16*, const __nv_bfloat16*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_f16(__half*, const __half*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_f32(float*, const float*, const uint32_t*, size_t, size_t);
extern "C" __global__ void moe_gather_u8(uint8_t*, const uint8_t*, const uint32_t*, size_t, size_t);
// Fused router: softmax + top-k select + (optional) renormalize, one thread per token
extern "C" __global__ void moe_route_f32(const float*, uint32_t*, float*, int, int, int, int);
extern "C" __global__ void moe_route_f16(const __half*, uint32_t*, float*, int, int, int, int);
extern "C" __global__ void moe_route_bf16(const __nv_bfloat16*, uint32_t*, float*, int, int, int, int);
// Deterministic scatter (sequential per-token reduce, no atomicAdd, variable k via prefix sum)
extern "C" __global__ void deterministic_scatter_bf16(__nv_bfloat16*, const __nv_bfloat16*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);
extern "C" __global__ void deterministic_scatter_f16(__half*, const __half*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);
extern "C" __global__ void deterministic_scatter_f32(float*, const float*, const uint32_t*, const float*, const uint32_t*, const int*, int, int);

// =============================================================================
// DISPATCHER FUNCTIONS
// =============================================================================

// Binary arithmetic dispatcher
// op: 0=add, 1=div, 2=mul, 3=sub, 4=minimum, 5=maximum
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
extern "C" void run_binary_arith_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* lhs, const void* rhs, void* out) {
    int grid = grid_size(numel);
    
    #define DISPATCH_ARITH(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 1: bdiv_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 2: bmul_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 3: bsub_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 4: bminimum_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 5: bmaximum_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
        }

    switch (dtype) {
        case 0: DISPATCH_ARITH(f32, float); break;
        case 1: DISPATCH_ARITH(f64, double); break;
        case 2: DISPATCH_ARITH(u8, uint8_t); break;
        case 3: DISPATCH_ARITH(u32, uint32_t); break;
        case 4: DISPATCH_ARITH(i64, int64_t); break;
        case 5: DISPATCH_ARITH(f16, __half); break;
        case 6: DISPATCH_ARITH(bf16, __nv_bfloat16); break;
        case 7: DISPATCH_ARITH(f8_e4m3, __nv_fp8_e4m3); break;
    }
    #undef DISPATCH_ARITH
}

// Binary comparison dispatcher
// op: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
extern "C" void run_binary_cmp_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* lhs, const void* rhs, uint8_t* out) {
    int grid = grid_size(numel);
    
    #define DISPATCH_CMP(DTYPE, CTYPE) \
        switch (op) { \
            case 0: eq_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 1: ne_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 2: lt_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 3: le_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 4: gt_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 5: ge_##DTYPE<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
        }

    switch (dtype) {
        case 0: DISPATCH_CMP(f32, float); break;
        case 1: DISPATCH_CMP(f64, double); break;
        case 2: DISPATCH_CMP(u8, uint8_t); break;
        case 3: DISPATCH_CMP(u32, uint32_t); break;
        case 4: DISPATCH_CMP(i64, int64_t); break;
        case 5: DISPATCH_CMP(f16, __half); break;
        case 6: DISPATCH_CMP(bf16, __nv_bfloat16); break;
        case 7: DISPATCH_CMP(f8_e4m3, __nv_fp8_e4m3); break;
    }
    #undef DISPATCH_CMP
}

// In-place binary arithmetic dispatcher
// op: 0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
// NOTE: lhs must be contiguous (we write back to it)
extern "C" void run_binary_inplace_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, void* lhs, const void* rhs) {
    int grid = grid_size(numel);
    
    // For float types, dispatch min/max separately (different naming)
    #define DISPATCH_INPLACE_FLOAT(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 1: bsub_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 2: bmul_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 3: bdiv_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 4: bmin_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 5: bmax_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
        }
    
    // For integer types (no min/max)
    #define DISPATCH_INPLACE_INT(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 1: bsub_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 2: bmul_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 3: bdiv_##DTYPE##_inplace<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
        }

    switch (dtype) {
        case 0: DISPATCH_INPLACE_FLOAT(f32, float); break;
        case 1: DISPATCH_INPLACE_FLOAT(f64, double); break;
        case 2: DISPATCH_INPLACE_INT(u8, uint8_t); break;
        case 3: DISPATCH_INPLACE_INT(u32, uint32_t); break;
        case 4: DISPATCH_INPLACE_INT(i64, int64_t); break;
        case 5: DISPATCH_INPLACE_FLOAT(f16, __half); break;
        case 6: DISPATCH_INPLACE_FLOAT(bf16, __nv_bfloat16); break;
        case 7: DISPATCH_INPLACE_FLOAT(f8_e4m3, __nv_fp8_e4m3); break;
    }
    #undef DISPATCH_INPLACE_FLOAT
    #undef DISPATCH_INPLACE_INT
}

// Parametric unary dispatcher (elu, powf)
// op: 0=elu, 1=powf
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16
void run_unary_param_op(int32_t op, int32_t dtype, float param, size_t numel, size_t num_dims, const size_t* info, const void* inp, void* out) {
    int grid = grid_size(numel);
    
    if (op == 0) { // elu
        switch (dtype) {
            case 0: uelu_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, param, (const float*)inp, (float*)out); break;
            case 1: uelu_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (double)param, (const double*)inp, (double*)out); break;
            case 2: uelu_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, __float2half(param), (const __half*)inp, (__half*)out); break;
            case 3: uelu_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, __float2bfloat16(param), (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
        }
    } else if (op == 1) { // powf
        switch (dtype) {
            case 0: upowf_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, param, (const float*)inp, (float*)out); break;
            case 1: upowf_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (double)param, (const double*)inp, (double*)out); break;
            case 2: upowf_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, __float2half(param), (const __half*)inp, (__half*)out); break;
            case 3: upowf_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, __float2bfloat16(param), (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
        }
    }
}

// Where (ternary) dispatcher
// cond_dtype: 0=i64, 1=u32, 2=u8
// data_dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
void run_where(int32_t cond_dtype, int32_t data_dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* cond, const void* t, const void* f, void* out) {
    int grid = grid_size(numel);
    
    if (cond_dtype == 0) { // i64
        switch (data_dtype) {
            case 0: where_i64_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_i64_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_i64_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_i64_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_i64_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_i64_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_i64_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    } else if (cond_dtype == 1) { // u32
        switch (data_dtype) {
            case 0: where_u32_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_u32_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_u32_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_u32_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_u32_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_u32_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_u32_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    } else if (cond_dtype == 2) { // u8
        switch (data_dtype) {
            case 0: where_u8_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_u8_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_u8_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_u8_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_u8_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_u8_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_u8_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    }
}

// op: 0=sum, 1=min, 2=max
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3
// REDUCE_SMEM_SIZE: Shared memory for block reductions (512 bytes is enough for 32 warps * 16 bytes)
constexpr int REDUCE_SMEM_SIZE = 512;

void run_fast_reduce_op(int32_t op, int32_t dtype, size_t src_numel, size_t el_per_reduce, size_t num_dims, const size_t* info, const void* src, void* dst) {
    int num_blocks = (src_numel + el_per_reduce - 1) / el_per_reduce;
    if (op == 0) { // Sum
        switch (dtype) {
            case 0: fast_sum_f32<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_sum_f64<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_sum_f16<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_sum_bf16<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_sum_u32<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_sum_i64<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_sum_u8<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_sum_f8_e4m3<<<num_blocks, BLOCK_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    } else if (op == 1) { // Min
        switch (dtype) {
            case 0: fast_min_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_min_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_min_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_min_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_min_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_min_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_min_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_min_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    } else if (op == 2) { // Max
        switch (dtype) {
            case 0: fast_max_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_max_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_max_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_max_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_max_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_max_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_max_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_max_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    }
}

// op: 0=argmin, 1=argmax
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3
void run_fast_arg_reduce_op(int32_t op, int32_t dtype, size_t src_numel, size_t el_per_reduce, size_t num_dims, const size_t* info, const void* src, uint32_t* dst) {
    int num_blocks = (src_numel + el_per_reduce - 1) / el_per_reduce;
    if (op == 0) { // ArgMin
        switch (dtype) {
            case 0: fast_argmin_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, dst); break;
            case 1: fast_argmin_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, dst); break;
            case 2: fast_argmin_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, dst); break;
            case 3: fast_argmin_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, dst); break;
            case 4: fast_argmin_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, dst); break;
            case 5: fast_argmin_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, dst); break;
            case 6: fast_argmin_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_argmin_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, dst); break;
#endif
        }
    } else if (op == 1) { // ArgMax
        switch (dtype) {
            case 0: fast_argmax_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, dst); break;
            case 1: fast_argmax_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, dst); break;
            case 2: fast_argmax_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, dst); break;
            case 3: fast_argmax_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, dst); break;
            case 4: fast_argmax_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, dst); break;
            case 5: fast_argmax_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, dst); break;
            case 6: fast_argmax_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_argmax_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, dst); break;
#endif
        }
    }
}

// Index select dispatcher
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_index_select(int32_t ids_dtype, int32_t dtype, size_t numel, size_t num_dims, const size_t* info, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size) {
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: is_i64_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_i64_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_i64_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_i64_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_i64_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_i64_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_i64_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: is_u32_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_u32_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_u32_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_u32_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_u32_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_u32_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_u32_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 4) { // u8
        switch (dtype) {
            case 0: is_u8_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_u8_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_u8_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_u8_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_u8_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_u8_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_u8_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    }
}

// Gather dispatcher  
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_gather(int32_t ids_dtype, int32_t dtype, size_t numel, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size) {
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: gather_i64_f32<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_i64_f64<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_i64_u8<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_i64_u32<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_i64_i64<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_i64_f16<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_i64_bf16<<<grid, BLOCK_SIZE>>>(numel, (const int64_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: gather_u32_f32<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_u32_f64<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_u32_u8<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_u32_u32<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_u32_i64<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_u32_f16<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_u32_bf16<<<grid, BLOCK_SIZE>>>(numel, (const uint32_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 4) { // u8
        switch (dtype) {
            case 0: gather_u8_f32<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_u8_f64<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_u8_u8<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_u8_u32<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_u8_i64<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_u8_f16<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_u8_bf16<<<grid, BLOCK_SIZE>>>(numel, (const uint8_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    }
}

// Index add dispatcher
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_index_add(int32_t ids_dtype, int32_t dtype, const void* ids, size_t ids_dim_size, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
    size_t numel = left_size * right_size;
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: ia_i64_f32<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, ids_dim_size, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 1: ia_i64_f64<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, ids_dim_size, (const double*)src, (double*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 5: ia_i64_f16<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, ids_dim_size, (const __half*)src, (__half*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 6: ia_i64_bf16<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, ids_dim_size, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: ia_u32_f32<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, ids_dim_size, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 1: ia_u32_f64<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, ids_dim_size, (const double*)src, (double*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 5: ia_u32_f16<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, ids_dim_size, (const __half*)src, (__half*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 6: ia_u32_bf16<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, ids_dim_size, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    }
}

// Scatter dispatcher
// dtype: 0=f32
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_scatter(int32_t ids_dtype, int32_t dtype, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
    size_t numel = left_size * right_size;
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: s_i64_f32<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: s_u32_f32<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    }
}

// Scatter add dispatcher
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_scatter_add(int32_t ids_dtype, int32_t dtype, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
    size_t numel = left_size * right_size;
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: sa_i64_f32<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 1: sa_i64_f64<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 5: sa_i64_f16<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 6: sa_i64_bf16<<<grid, BLOCK_SIZE>>>((const int64_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: sa_u32_f32<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 1: sa_u32_f64<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 5: sa_u32_f16<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
            case 6: sa_u32_bf16<<<grid, BLOCK_SIZE>>>((const uint32_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, dst_dim_size, right_size); break;
        }
    }
}

// Convolution-related run_* functions moved to conv_dispatcher.cu to avoid duplicate symbols
// Cast dispatcher run_cast moved to cast_dispatcher.cu

// Unary operation dispatcher
// op: 0=copy, 1=neg, 2=recip, 3=exp, 4=log, 5=sin, 6=cos, 7=tanh, 8=erf, 9=ceil, 10=floor, 
//     11=round, 12=normcdf, 13=abs, 14=sqr, 15=sqrt, 16=gelu, 17=gelu_erf, 18=relu, 19=silu, 20=sign, 21=sigmoid
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64 (only for copy)
void run_unary_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* info, const void* inp, void* out) {
    int grid = grid_size(numel);
    
    // Handle copy operation separately as it supports more dtypes
    if (op == 0) { // copy
        switch (dtype) {
            case 0: ucopy_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const float*)inp, (float*)out); break;
            case 1: ucopy_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const double*)inp, (double*)out); break;
            case 2: ucopy_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const __half*)inp, (__half*)out); break;
            case 3: ucopy_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
            case 4: ucopy_f8_e4m3<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (__nv_fp8_e4m3*)out); break;
            case 5: ucopy_u8<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint8_t*)inp, (uint8_t*)out); break;
            case 6: ucopy_u32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const uint32_t*)inp, (uint32_t*)out); break;
            case 7: ucopy_i64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const int64_t*)inp, (int64_t*)out); break;
        }
        return;
    }
    
    // For all other unary ops, only float types are supported
    #define DISPATCH_UNARY_F32(NAME) NAME##_f32<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const float*)inp, (float*)out)
    #define DISPATCH_UNARY_F64(NAME) NAME##_f64<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const double*)inp, (double*)out)
    #define DISPATCH_UNARY_F16(NAME) NAME##_f16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const __half*)inp, (__half*)out)
    #define DISPATCH_UNARY_BF16(NAME) NAME##_bf16<<<grid, BLOCK_SIZE>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out)
    
    #define DISPATCH_UNARY_ALL(NAME) \
        switch (dtype) { \
            case 0: DISPATCH_UNARY_F32(NAME); break; \
            case 1: DISPATCH_UNARY_F64(NAME); break; \
            case 2: DISPATCH_UNARY_F16(NAME); break; \
            case 3: DISPATCH_UNARY_BF16(NAME); break; \
        }
    
    switch (op) {
        case 1: DISPATCH_UNARY_ALL(uneg); break;
        case 2: DISPATCH_UNARY_ALL(urecip); break;
        case 3: DISPATCH_UNARY_ALL(uexp); break;
        case 4: DISPATCH_UNARY_ALL(ulog); break;
        case 5: DISPATCH_UNARY_ALL(usin); break;
        case 6: DISPATCH_UNARY_ALL(ucos); break;
        case 7: DISPATCH_UNARY_ALL(utanh); break;
        case 8: DISPATCH_UNARY_ALL(uerf); break;
        case 9: DISPATCH_UNARY_ALL(uceil); break;
        case 10: DISPATCH_UNARY_ALL(ufloor); break;
        case 11: DISPATCH_UNARY_ALL(uround); break;
        case 12: DISPATCH_UNARY_ALL(unormcdf); break;
        case 13: DISPATCH_UNARY_ALL(uabs); break;
        case 14: DISPATCH_UNARY_ALL(usqr); break;
        case 15: DISPATCH_UNARY_ALL(usqrt); break;
        case 16: DISPATCH_UNARY_ALL(ugelu); break;
        case 17: DISPATCH_UNARY_ALL(ugelu_erf); break;
        case 18: DISPATCH_UNARY_ALL(urelu); break;
        case 19: DISPATCH_UNARY_ALL(usilu); break;
        case 20: DISPATCH_UNARY_ALL(usign); break;
        case 21: DISPATCH_UNARY_ALL(usigmoid); break;
    }
    
    #undef DISPATCH_UNARY_F32
    #undef DISPATCH_UNARY_F64
    #undef DISPATCH_UNARY_F16
    #undef DISPATCH_UNARY_BF16
    #undef DISPATCH_UNARY_ALL
}

// Fused SiLU-Mul dispatcher
// dtype: 0=f32, 1=f16, 2=bf16, 3=f8_e4m3
// Computes: out[i] = silu(gate[i]) * up[i]
void run_fused_silu_mul(int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* gate, const void* up, void* out) {
    int grid = grid_size(numel);

    switch (dtype) {
        case 0: // f32
            fused_silu_mul_f32_vec4<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const float*)gate, (const float*)up, (float*)out);
            break;
        case 1: // f16
            fused_silu_mul_f16_vec2<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __half*)gate, (const __half*)up, (__half*)out);
            break;
        case 2: // bf16
            fused_silu_mul_bf16_vec2<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (__nv_bfloat16*)out);
            break;
        case 3: // f8_e4m3
            fused_silu_mul_f8_e4m3_vec4<<<grid, BLOCK_SIZE>>>(numel, num_dims, dims_and_strides, (const __nv_fp8_e4m3*)gate, (const __nv_fp8_e4m3*)up, (__nv_fp8_e4m3*)out);
            break;
    }
}

// Fused SwiGLU → q8a128 (producer epilogue B4): out = quantize(silu(gate)*up).
extern "C" __global__ void silu_mul_q8a128_f32(const float*, const float*, void*, int, int);
extern "C" __global__ void silu_mul_q8a128_f16(const __half*, const __half*, void*, int, int);
extern "C" __global__ void silu_mul_q8a128_bf16(const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int);

void run_silu_mul_q8a128_op(int32_t dtype, const void* gate, const void* up, void* out, int rows, int cols) {
    long long total_tiles = ((long long)rows * cols) / 128;
    if (total_tiles <= 0) return;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    long long blocks = (total_tiles + warps_per_block - 1) / warps_per_block;
    if (blocks > 65535) blocks = 65535; // grid-stride covers the remainder
    dim3 grid((unsigned)blocks, 1, 1), block(threads, 1, 1);
    switch (dtype) {
        case 0: // f32
            silu_mul_q8a128_f32<<<grid, block>>>((const float*)gate, (const float*)up, out, rows, cols);
            break;
        case 1: // f16
            silu_mul_q8a128_f16<<<grid, block>>>((const __half*)gate, (const __half*)up, out, rows, cols);
            break;
        case 2: // bf16
            silu_mul_q8a128_bf16<<<grid, block>>>((const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, out, rows, cols);
            break;
    }
}

// =============================================================================
// FUSED MoE GATHER / WEIGHTED SCATTER-ADD
// =============================================================================
// Grid: (total_rows, ceil(hidden_dim / BLOCK_SIZE)) — 2D grid

void run_moe_gather(int32_t dtype, void* out, const void* xs,
                    const uint32_t* token_ids,
                    size_t total_rows, size_t hidden_dim) {
    if (total_rows == 0) return;
    dim3 grid(total_rows, grid_size(hidden_dim));
    switch (dtype) {
        case 0: // f32
            moe_gather_f32<<<grid, BLOCK_SIZE>>>((float*)out, (const float*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 1: // f16
            moe_gather_f16<<<grid, BLOCK_SIZE>>>((__half*)out, (const __half*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 2: // bf16
            moe_gather_bf16<<<grid, BLOCK_SIZE>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 3: // u8 (q8a1024 byte-row gather; hidden_dim = per-token byte count)
            moe_gather_u8<<<grid, BLOCK_SIZE>>>((uint8_t*)out, (const uint8_t*)xs, token_ids, total_rows, hidden_dim);
            break;
    }
}

// Fused router. One thread per token (grid-stride). `logits` is [num_tokens, n_experts]
// in `dtype` (0=f32,1=f16,2=bf16); writes top-k expert indices (u32) and routing weights
// (f32), both [num_tokens, k] in descending-logit order. `norm_topk` selects renormalized
// top-k softmax (1) vs plain full-softmax weights (0).
void run_moe_route(int32_t dtype, const void* logits, uint32_t* out_idx, float* out_weights,
                   int num_tokens, int n_experts, int k, int norm_topk) {
    if (num_tokens == 0) return;
    // One warp per token; 256-thread blocks pack 8 warps (= 8 tokens) each.
    const int tpb = 256;
    const int warps_per_block = tpb / 32;
    const int blocks = (num_tokens + warps_per_block - 1) / warps_per_block;
    switch (dtype) {
        case 0: // f32
            moe_route_f32<<<blocks, tpb>>>((const float*)logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
            break;
        case 1: // f16
            moe_route_f16<<<blocks, tpb>>>((const __half*)logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
            break;
        case 2: // bf16
            moe_route_bf16<<<blocks, tpb>>>((const __nv_bfloat16*)logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
            break;
    }
}

// Deterministic MoE scatter: sequential per-token reduce, no atomicAdd.
// perm[i] maps token-major index i to the expert-major row in down_out.
// grid: (num_tokens, ceil(hidden_dim / BLOCK_SIZE))
void run_deterministic_scatter(int32_t dtype, void* ys, const void* down_out,
                               const uint32_t* perm,
                               const float* weights_flat,
                               const uint32_t* reordered_weight_ids,
                               const int* token_starts,
                               int num_tokens, int hidden) {
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, grid_size(hidden));
    switch (dtype) {
        case 0: // f32
            deterministic_scatter_f32<<<grid, BLOCK_SIZE>>>(
                (float*)ys, (const float*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
        case 1: // f16
            deterministic_scatter_f16<<<grid, BLOCK_SIZE>>>(
                (__half*)ys, (const __half*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
        case 2: // bf16
            deterministic_scatter_bf16<<<grid, BLOCK_SIZE>>>(
                (__nv_bfloat16*)ys, (const __nv_bfloat16*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
    }
}

}  // extern "C"
