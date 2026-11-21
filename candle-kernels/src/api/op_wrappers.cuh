// `extern "C"` wrappers for affine, fill, binary, unary, reduce, softmax,
// where, index, sort, cast and multinomial, plus integer and FP8 binary
// arithmetic.
//
// The `run_*` entry points for the core elementwise and reduction families,
// followed by the `badd_`/`bmul_`/`bdiv_`/`bsub_`/`bminimum_`/`bmaximum_`
// wrappers for the u8, u32, i64, f64, f16, bf16 and f8_e4m3 dtypes.
// They are the first block of the C API that Rust calls through FFI.

// --- Affine wrappers ---
void run_affine_f32(
    const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add,
    void* stream
) {
    affine_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_f64(
    const double* inp, double* out,
    size_t numel, size_t num_dims, const size_t* info,
    double mul, double add,
    void* stream
) {
    affine_f64<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_f16(
    const void* inp, void* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add,
    void* stream
) {
    __half h_mul = __float2half(mul);
    __half h_add = __float2half(add);
    affine_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __half*)inp, (__half*)out, h_mul, h_add);
}

void run_affine_bf16(
    const void* inp, void* out,
    size_t numel, size_t num_dims, const size_t* info,
    float mul, float add,
    void* stream
) {
    __nv_bfloat16 h_mul = __float2bfloat16(mul);
    __nv_bfloat16 h_add = __float2bfloat16(add);
    affine_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out, h_mul, h_add);
}

void run_affine_u8(
    const uint8_t* inp, uint8_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    uint8_t mul, uint8_t add,
    void* stream
) {
    affine_u8<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_u32(
    const uint32_t* inp, uint32_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    uint32_t mul, uint32_t add,
    void* stream
) {
    affine_u32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i16(
    const int16_t* inp, int16_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int16_t mul, int16_t add,
    void* stream
) {
    affine_i16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i32(
    const int32_t* inp, int32_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int32_t mul, int32_t add,
    void* stream
) {
    affine_i32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

void run_affine_i64(
    const int64_t* inp, int64_t* out,
    size_t numel, size_t num_dims, const size_t* info,
    int64_t mul, int64_t add,
    void* stream
) {
    affine_i64<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out, mul, add);
}

// --- Fill wrappers ---
void run_fill_f32(float* buf, float value, size_t numel, void* stream) {
    fill_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(buf, value, numel);
}

void run_fill_f64(double* buf, double value, size_t numel, void* stream) {
    fill_f64<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(buf, value, numel);
}

void run_fill_f16(void* buf, float value, size_t numel, void* stream) {
    __half h_val = __float2half(value);
    fill_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>((__half*)buf, h_val, numel);
}

void run_fill_bf16(void* buf, float value, size_t numel, void* stream) {
    __nv_bfloat16 h_val = __float2bfloat16(value);
    fill_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>((__nv_bfloat16*)buf, h_val, numel);
}

void run_fill_u8(uint8_t* buf, uint8_t value, size_t numel, void* stream) {
    fill_u8<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(buf, value, numel);
}

void run_fill_u32(uint32_t* buf, uint32_t value, size_t numel, void* stream) {
    fill_u32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(buf, value, numel);
}

void run_fill_i64(int64_t* buf, int64_t value, size_t numel, void* stream) {
    fill_i64<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(buf, value, numel);
}

// --- Binary operation wrappers ---
void run_binary_f32(
    const float* lhs, const float* rhs, float* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op,  // 0=add, 1=mul, 2=div, 3=sub, 4=min, 5=max
    void* stream
) {
    switch (op) {
        case 0: badd_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 1: bmul_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 2: bdiv_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 3: bsub_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 4: bminimum_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
        case 5: bmaximum_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, lhs, rhs, out); break;
    }
}

void run_binary_f16(
    const void* lhs, const void* rhs, void* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op,
    void* stream
) {
    switch (op) {
        case 0: badd_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 1: bmul_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 2: bdiv_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
        case 3: bsub_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __half*)lhs, (const __half*)rhs, (__half*)out); break;
    }
}

void run_binary_bf16(
    const void* lhs, const void* rhs, void* out,
    size_t numel, size_t num_dims, const size_t* dims_and_strides,
    int op,
    void* stream
) {
    switch (op) {
        case 0: badd_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 1: bmul_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 2: bdiv_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
        case 3: bsub_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)lhs, (const __nv_bfloat16*)rhs, (__nv_bfloat16*)out); break;
    }
}

// --- Unary operation wrappers ---
void run_unary_f32(
    const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    int op,  // 0=copy, 1=neg, 2=exp, 3=log, 4=sin, 5=cos, 6=tanh, 7=sqrt, 8=gelu, 9=relu, 10=silu, 11=sigmoid
    void* stream
) {
    switch (op) {
        case 0: ucopy_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 1: uneg_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 2: uexp_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 3: ulog_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 4: usin_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 5: ucos_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 6: utanh_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 7: usqrt_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 8: ugelu_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 9: urelu_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 10: usilu_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
        case 11: usigmoid_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, out); break;
    }
}

// --- Reduce wrappers ---
void run_reduce_sum_f32(
    const float* src, float* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info,
    void* stream
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_sum_f32<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_reduce_max_f32(
    const float* src, float* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info,
    void* stream
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_max_f32<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_reduce_argmax_f32(
    const float* src, uint32_t* dst,
    size_t src_numel, size_t el_per_block,
    size_t num_dims, const size_t* info,
    void* stream
) {
    int num_blocks = (src_numel + el_per_block - 1) / el_per_block;
    fast_argmax_f32<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_block, num_dims, info, src, dst);
}

void run_softmax_f32(const float* src, float* dst, int ncols, int nrows, void* stream) {
    softmax_f32<<<nrows, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src, dst, ncols, nrows);
}

// --- Where (ternary) wrappers ---
void run_where_u8_f32(
    const uint8_t* ids, const float* t, const float* f, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    void* stream
) {
    where_u8_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, ids, t, f, out);
}

// --- Index select wrapper ---
void run_index_select_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t numel, size_t num_dims, const size_t* info,
    size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size,
    void* stream
) {
    is_u32_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size);
}

// --- Gather wrapper ---
void run_gather_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t numel, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size,
    void* stream
) {
    gather_u32_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size);
}

// --- Index add wrapper ---
void run_index_add_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t ids_dim_size, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size,
    void* stream
) {
    size_t numel = left_size * right_size;
    ia_u32_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size);
}

// --- Scatter add wrapper ---
void run_scatter_add_u32_f32(
    const uint32_t* ids, const float* inp, float* out,
    size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size,
    void* stream
) {
    size_t numel = left_size * right_size;
    sa_u32_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size);
}

// --- Sort wrappers ---
void run_argsort_asc_f32(const float* src, uint32_t* dst, size_t nrows, size_t ncols, size_t ncols_pad, void* stream) {
    asort_asc_f32<<<nrows, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src, dst, nrows, ncols, ncols_pad);
}

void run_argsort_desc_f32(const float* src, uint32_t* dst, size_t nrows, size_t ncols, size_t ncols_pad, void* stream) {
    asort_desc_f32<<<nrows, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src, dst, nrows, ncols, ncols_pad);
}

// --- Cast wrappers ---
void run_cast_f32_f16(const float* inp, void* out, size_t numel, size_t num_dims, const size_t* info, void* stream) {
    cast_f32_f16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, (__half*)out);
}

void run_cast_f16_f32(const void* inp, float* out, size_t numel, size_t num_dims, const size_t* info, void* stream) {
    cast_f16_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __half*)inp, out);
}

void run_cast_f32_bf16(const float* inp, void* out, size_t numel, size_t num_dims, const size_t* info, void* stream) {
    cast_f32_bf16<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, inp, (__nv_bfloat16*)out);
}

void run_cast_bf16_f32(const void* inp, float* out, size_t numel, size_t num_dims, const size_t* info, void* stream) {
    cast_bf16_f32<<<grid_size(numel), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, out);
}

// NOTE: run_quantize_q8_1 and run_dequantize_mul_mat_vec are defined in quantized_dispatcher.cu

// --- Multinomial wrapper ---
void run_multinomial_f32(
    const float* logits, uint32_t* output, float* workspace,
    uint32_t vocab_size, uint32_t top_k, float top_p, float temperature, uint64_t seed,
    void* stream
) {
    optimized_multinomial_f32<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), (cudaStream_t)stream>>>(
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
