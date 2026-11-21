// Launch constants, grid sizing and the forward declarations of the simple
// `__global__` kernels.
//
// Holds `BLOCK_SIZE`, the `grid_size` helper and the prototypes of every simple
// kernel family the wrappers launch: affine, fill, binary, unary, reduce,
// index, where, convolution, pooling, sort, cast, quantized matmul and
// multinomial.
// They are declared before the `extern "C"` wrappers so each wrapper names a
// kernel already in scope.

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
