// Dtype-and-op dispatcher functions for every wrapper family, with the
// scatter-family macros they share.
//
// The `run_*` dispatchers that select a kernel by dtype and op for arithmetic,
// comparison, in-place, parametric unary, where, reduce, index, gather,
// scatter, unary, fused SiLU-mul and MoE work.
// The `SCATTER_FAMILY_*` macros are defined and undefined inside this run, so
// they stay with the dispatchers that use them.

// =============================================================================
// DISPATCHER FUNCTIONS
// =============================================================================

// Binary arithmetic dispatcher
// op: 0=add, 1=div, 2=mul, 3=sub, 4=minimum, 5=maximum
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
extern "C" void run_binary_arith_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* lhs, const void* rhs, void* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    #define DISPATCH_ARITH(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 1: bdiv_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 2: bmul_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 3: bsub_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 4: bminimum_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
            case 5: bmaximum_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, (CTYPE*)out); break; \
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
extern "C" void run_binary_cmp_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* lhs, const void* rhs, uint8_t* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    #define DISPATCH_CMP(DTYPE, CTYPE) \
        switch (op) { \
            case 0: eq_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 1: ne_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 2: lt_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 3: le_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 4: gt_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
            case 5: ge_##DTYPE<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const CTYPE*)lhs, (const CTYPE*)rhs, out); break; \
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
extern "C" void run_binary_inplace_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, void* lhs, const void* rhs, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    // For float types, dispatch min/max separately (different naming)
    #define DISPATCH_INPLACE_FLOAT(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 1: bsub_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 2: bmul_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 3: bdiv_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 4: bmin_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 5: bmax_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
        }
    
    // For integer types (no min/max)
    #define DISPATCH_INPLACE_INT(DTYPE, CTYPE) \
        switch (op) { \
            case 0: badd_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 1: bsub_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 2: bmul_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
            case 3: bdiv_##DTYPE##_inplace<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (CTYPE*)lhs, (const CTYPE*)rhs); break; \
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
void run_unary_param_op(int32_t op, int32_t dtype, float param, size_t numel, size_t num_dims, const size_t* info, const void* inp, void* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    if (op == 0) { // elu
        switch (dtype) {
            case 0: uelu_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, param, (const float*)inp, (float*)out); break;
            case 1: uelu_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (double)param, (const double*)inp, (double*)out); break;
            case 2: uelu_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, __float2half(param), (const __half*)inp, (__half*)out); break;
            case 3: uelu_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, __float2bfloat16(param), (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
        }
    } else if (op == 1) { // powf
        switch (dtype) {
            case 0: upowf_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, param, (const float*)inp, (float*)out); break;
            case 1: upowf_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (double)param, (const double*)inp, (double*)out); break;
            case 2: upowf_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, __float2half(param), (const __half*)inp, (__half*)out); break;
            case 3: upowf_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, __float2bfloat16(param), (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
        }
    }
}

// Where (ternary) dispatcher
// cond_dtype: 0=i64, 1=u32, 2=u8
// data_dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
void run_where(int32_t cond_dtype, int32_t data_dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* cond, const void* t, const void* f, void* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    if (cond_dtype == 0) { // i64
        switch (data_dtype) {
            case 0: where_i64_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_i64_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_i64_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_i64_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_i64_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_i64_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_i64_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const int64_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    } else if (cond_dtype == 1) { // u32
        switch (data_dtype) {
            case 0: where_u32_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_u32_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_u32_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_u32_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_u32_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_u32_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_u32_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint32_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    } else if (cond_dtype == 2) { // u8
        switch (data_dtype) {
            case 0: where_u8_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const float*)t, (const float*)f, (float*)out); break;
            case 1: where_u8_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const double*)t, (const double*)f, (double*)out); break;
            case 2: where_u8_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const uint8_t*)t, (const uint8_t*)f, (uint8_t*)out); break;
            case 3: where_u8_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const uint32_t*)t, (const uint32_t*)f, (uint32_t*)out); break;
            case 4: where_u8_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const int64_t*)t, (const int64_t*)f, (int64_t*)out); break;
            case 5: where_u8_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const __half*)t, (const __half*)f, (__half*)out); break;
            case 6: where_u8_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const uint8_t*)cond, (const __nv_bfloat16*)t, (const __nv_bfloat16*)f, (__nv_bfloat16*)out); break;
        }
    }
}

// op: 0=sum, 1=min, 2=max
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3
// REDUCE_SMEM_SIZE: Shared memory for block reductions (512 bytes is enough for 32 warps * 16 bytes)
constexpr int REDUCE_SMEM_SIZE = 512;

void run_fast_reduce_op(int32_t op, int32_t dtype, size_t src_numel, size_t el_per_reduce, size_t num_dims, const size_t* info, const void* src, void* dst, void* stream) {
    int num_blocks = (src_numel + el_per_reduce - 1) / el_per_reduce;
    if (op == 0) { // Sum
        switch (dtype) {
            case 0: fast_sum_f32<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_sum_f64<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_sum_f16<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_sum_bf16<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_sum_u32<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_sum_i64<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_sum_u8<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_sum_f8_e4m3<<<num_blocks, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    } else if (op == 1) { // Min
        switch (dtype) {
            case 0: fast_min_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_min_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_min_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_min_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_min_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_min_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_min_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_min_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    } else if (op == 2) { // Max
        switch (dtype) {
            case 0: fast_max_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, (float*)dst); break;
            case 1: fast_max_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, (double*)dst); break;
            case 2: fast_max_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, (__half*)dst); break;
            case 3: fast_max_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst); break;
            case 4: fast_max_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, (uint32_t*)dst); break;
            case 5: fast_max_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, (int64_t*)dst); break;
            case 6: fast_max_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, (uint8_t*)dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_max_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, (__nv_fp8_e4m3*)dst); break;
#endif
        }
    }
}

// op: 0=argmin, 1=argmax
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3
void run_fast_arg_reduce_op(int32_t op, int32_t dtype, size_t src_numel, size_t el_per_reduce, size_t num_dims, const size_t* info, const void* src, uint32_t* dst, void* stream) {
    int num_blocks = (src_numel + el_per_reduce - 1) / el_per_reduce;
    if (op == 0) { // ArgMin
        switch (dtype) {
            case 0: fast_argmin_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, dst); break;
            case 1: fast_argmin_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, dst); break;
            case 2: fast_argmin_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, dst); break;
            case 3: fast_argmin_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, dst); break;
            case 4: fast_argmin_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, dst); break;
            case 5: fast_argmin_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, dst); break;
            case 6: fast_argmin_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_argmin_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, dst); break;
#endif
        }
    } else if (op == 1) { // ArgMax
        switch (dtype) {
            case 0: fast_argmax_f32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const float*)src, dst); break;
            case 1: fast_argmax_f64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const double*)src, dst); break;
            case 2: fast_argmax_f16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __half*)src, dst); break;
            case 3: fast_argmax_bf16<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_bfloat16*)src, dst); break;
            case 4: fast_argmax_u32<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint32_t*)src, dst); break;
            case 5: fast_argmax_i64<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const int64_t*)src, dst); break;
            case 6: fast_argmax_u8<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const uint8_t*)src, dst); break;
#if __CUDA_ARCH__ >= 890 || !defined(__CUDA_ARCH__)
            case 7: fast_argmax_f8_e4m3<<<num_blocks, BLOCK_SIZE, REDUCE_SMEM_SIZE, (cudaStream_t)stream>>>(src_numel, el_per_reduce, num_dims, info, (const __nv_fp8_e4m3*)src, dst); break;
#endif
        }
    }
}

// Index select dispatcher
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_index_select(int32_t ids_dtype, int32_t dtype, size_t numel, size_t num_dims, const size_t* info, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: is_i64_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_i64_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_i64_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_i64_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_i64_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_i64_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_i64_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: is_u32_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_u32_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_u32_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_u32_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_u32_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_u32_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_u32_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 4) { // u8
        switch (dtype) {
            case 0: is_u8_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: is_u8_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: is_u8_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: is_u8_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: is_u8_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: is_u8_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: is_u8_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    }
}

// Gather dispatcher  
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16
// ids_dtype: 0=i16, 1=i32, 2=i64, 3=u32, 4=u8
void run_gather(int32_t ids_dtype, int32_t dtype, size_t numel, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    if (ids_dtype == 2) { // i64
        switch (dtype) {
            case 0: gather_i64_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_i64_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_i64_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_i64_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_i64_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_i64_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_i64_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const int64_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 3) { // u32
        switch (dtype) {
            case 0: gather_u32_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_u32_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_u32_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_u32_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_u32_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_u32_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_u32_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint32_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    } else if (ids_dtype == 4) { // u8
        switch (dtype) {
            case 0: gather_u8_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const float*)src, (float*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 1: gather_u8_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const double*)src, (double*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 2: gather_u8_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const uint8_t*)src, (uint8_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 3: gather_u8_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const uint32_t*)src, (uint32_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 4: gather_u8_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const int64_t*)src, (int64_t*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 5: gather_u8_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const __half*)src, (__half*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
            case 6: gather_u8_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, (const uint8_t*)ids, (const __nv_bfloat16*)src, (__nv_bfloat16*)dst, left_size, src_dim_size, ids_dim_size, right_size); break;
        }
    }
}

// The scatter-family dispatchers below return 0 when they launched a kernel
// and -1 when no kernel exists for the (ids_dtype, dtype) pair. The caller
// turns -1 into an error: a dispatcher that fell through silently reported a
// write that never happened as done.
//
// dtype: 0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3
// ids_dtype: 2=i64, 3=u32, 4=u8
#define SCATTER_FAMILY_SWITCH(LAUNCH, PREFIX, I)                     \
    switch (dtype) {                                                 \
        case 0: LAUNCH(PREFIX##_f32, I, float) return 0;             \
        case 1: LAUNCH(PREFIX##_f64, I, double) return 0;            \
        case 2: LAUNCH(PREFIX##_u8, I, uint8_t) return 0;            \
        case 3: LAUNCH(PREFIX##_u32, I, uint32_t) return 0;          \
        case 4: LAUNCH(PREFIX##_i64, I, int64_t) return 0;           \
        case 5: LAUNCH(PREFIX##_f16, I, __half) return 0;            \
        case 6: LAUNCH(PREFIX##_bf16, I, __nv_bfloat16) return 0;    \
    }

// The accumulating ops also carry an FP8 kernel per index type; scatter does not.
#define SCATTER_FAMILY_F8(LAUNCH, PREFIX, I)                         \
    if (dtype == 7) {                                                \
        LAUNCH(PREFIX##_f8_e4m3, I, __nv_fp8_e4m3) return 0;         \
    }

// Index add dispatcher
int32_t run_index_add(int32_t ids_dtype, int32_t dtype, const void* ids, size_t ids_dim_size, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size, void* stream) {
    size_t numel = left_size * right_size;
    if (numel == 0) return 0;
    int grid = grid_size(numel);
#define IA_LAUNCH(FN, I, T) FN<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>((const I*)ids, ids_dim_size, (const T*)src, (T*)dst, left_size, src_dim_size, dst_dim_size, right_size);
    if (ids_dtype == 2) {
        SCATTER_FAMILY_SWITCH(IA_LAUNCH, ia_i64, int64_t)
        SCATTER_FAMILY_F8(IA_LAUNCH, ia_i64, int64_t)
    } else if (ids_dtype == 3) {
        SCATTER_FAMILY_SWITCH(IA_LAUNCH, ia_u32, uint32_t)
        SCATTER_FAMILY_F8(IA_LAUNCH, ia_u32, uint32_t)
    } else if (ids_dtype == 4) {
        SCATTER_FAMILY_SWITCH(IA_LAUNCH, ia_u8, uint8_t)
        SCATTER_FAMILY_F8(IA_LAUNCH, ia_u8, uint8_t)
    }
#undef IA_LAUNCH
    return -1;
}

// Scatter dispatcher
int32_t run_scatter(int32_t ids_dtype, int32_t dtype, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size, void* stream) {
    size_t numel = left_size * right_size;
    if (numel == 0) return 0;
    int grid = grid_size(numel);
#define S_LAUNCH(FN, I, T) FN<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>((const I*)ids, (const T*)src, (T*)dst, left_size, src_dim_size, dst_dim_size, right_size);
    if (ids_dtype == 2) {
        SCATTER_FAMILY_SWITCH(S_LAUNCH, s_i64, int64_t)
    } else if (ids_dtype == 3) {
        SCATTER_FAMILY_SWITCH(S_LAUNCH, s_u32, uint32_t)
    } else if (ids_dtype == 4) {
        SCATTER_FAMILY_SWITCH(S_LAUNCH, s_u8, uint8_t)
    }
#undef S_LAUNCH
    return -1;
}

// Scatter add dispatcher
int32_t run_scatter_add(int32_t ids_dtype, int32_t dtype, const void* ids, const void* src, void* dst, size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size, void* stream) {
    size_t numel = left_size * right_size;
    if (numel == 0) return 0;
    int grid = grid_size(numel);
#define SA_LAUNCH(FN, I, T) FN<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>((const I*)ids, (const T*)src, (T*)dst, left_size, src_dim_size, dst_dim_size, right_size);
    if (ids_dtype == 2) {
        SCATTER_FAMILY_SWITCH(SA_LAUNCH, sa_i64, int64_t)
        SCATTER_FAMILY_F8(SA_LAUNCH, sa_i64, int64_t)
    } else if (ids_dtype == 3) {
        SCATTER_FAMILY_SWITCH(SA_LAUNCH, sa_u32, uint32_t)
        SCATTER_FAMILY_F8(SA_LAUNCH, sa_u32, uint32_t)
    } else if (ids_dtype == 4) {
        SCATTER_FAMILY_SWITCH(SA_LAUNCH, sa_u8, uint8_t)
        SCATTER_FAMILY_F8(SA_LAUNCH, sa_u8, uint8_t)
    }
#undef SA_LAUNCH
    return -1;
}

#undef SCATTER_FAMILY_F8
#undef SCATTER_FAMILY_SWITCH

// Convolution-related run_* functions moved to conv_dispatcher.cu to avoid duplicate symbols
// Cast dispatcher run_cast moved to cast_dispatcher.cu

// Unary operation dispatcher
// op: 0=copy, 1=neg, 2=recip, 3=exp, 4=log, 5=sin, 6=cos, 7=tanh, 8=erf, 9=ceil, 10=floor, 
//     11=round, 12=normcdf, 13=abs, 14=sqr, 15=sqrt, 16=gelu, 17=gelu_erf, 18=relu, 19=silu, 20=sign, 21=sigmoid
// dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3, 5=u8, 6=u32, 7=i64 (only for copy)
void run_unary_op(int32_t op, int32_t dtype, size_t numel, size_t num_dims, const size_t* info, const void* inp, void* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);
    
    // Handle copy operation separately as it supports more dtypes
    if (op == 0) { // copy
        switch (dtype) {
            case 0: ucopy_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const float*)inp, (float*)out); break;
            case 1: ucopy_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const double*)inp, (double*)out); break;
            case 2: ucopy_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __half*)inp, (__half*)out); break;
            case 3: ucopy_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out); break;
            case 4: ucopy_f8_e4m3<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __nv_fp8_e4m3*)inp, (__nv_fp8_e4m3*)out); break;
            case 5: ucopy_u8<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint8_t*)inp, (uint8_t*)out); break;
            case 6: ucopy_u32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const uint32_t*)inp, (uint32_t*)out); break;
            case 7: ucopy_i64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const int64_t*)inp, (int64_t*)out); break;
        }
        return;
    }
    
    // For all other unary ops, only float types are supported
    #define DISPATCH_UNARY_F32(NAME) NAME##_f32<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const float*)inp, (float*)out)
    #define DISPATCH_UNARY_F64(NAME) NAME##_f64<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const double*)inp, (double*)out)
    #define DISPATCH_UNARY_F16(NAME) NAME##_f16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __half*)inp, (__half*)out)
    #define DISPATCH_UNARY_BF16(NAME) NAME##_bf16<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, info, (const __nv_bfloat16*)inp, (__nv_bfloat16*)out)
    
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
void run_fused_silu_mul(int32_t dtype, size_t numel, size_t num_dims, const size_t* dims_and_strides, const void* gate, const void* up, void* out, void* stream) {
    if (numel == 0) return;
    int grid = grid_size(numel);

    switch (dtype) {
        case 0: // f32
            fused_silu_mul_f32_vec4<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const float*)gate, (const float*)up, (float*)out);
            break;
        case 1: // f16
            fused_silu_mul_f16_vec2<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __half*)gate, (const __half*)up, (__half*)out);
            break;
        case 2: // bf16
            fused_silu_mul_bf16_vec2<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (__nv_bfloat16*)out);
            break;
        case 3: // f8_e4m3
            fused_silu_mul_f8_e4m3_vec4<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(numel, num_dims, dims_and_strides, (const __nv_fp8_e4m3*)gate, (const __nv_fp8_e4m3*)up, (__nv_fp8_e4m3*)out);
            break;
    }
}

// Fused SwiGLU → q8a128 (producer epilogue B4): out = quantize(silu(gate)*up).
extern "C" __global__ void silu_mul_q8a128_f32(const float*, const float*, void*, int, int, int, int);
extern "C" __global__ void silu_mul_q8a128_f16(const __half*, const __half*, void*, int, int, int, int);
extern "C" __global__ void silu_mul_q8a128_bf16(const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int, int);

// `sum_norm` is `SumScale::as_code()` — the Σx convention these blocks are
// written in, which the matmul reading them must be launched with.
// `row_stride`: elements between consecutive rows of `gate`/`up` — `cols` when dense.
void run_silu_mul_q8a128_op(int32_t dtype, const void* gate, const void* up, void* out, int rows, int cols, int sum_norm, int row_stride, cudaStream_t stream) {
    long long total_tiles = ((long long)rows * cols) / 128;
    if (total_tiles <= 0) return;
    const int threads = 256;
    const int warps_per_block = threads / 32;
    long long blocks = (total_tiles + warps_per_block - 1) / warps_per_block;
    if (blocks > 65535) blocks = 65535; // grid-stride covers the remainder
    dim3 grid((unsigned)blocks, 1, 1), block(threads, 1, 1);
    switch (dtype) {
        case 0: // f32
            silu_mul_q8a128_f32<<<grid, block, 0, stream>>>((const float*)gate, (const float*)up, out, rows, cols, sum_norm, row_stride);
            break;
        case 1: // f16
            silu_mul_q8a128_f16<<<grid, block, 0, stream>>>((const __half*)gate, (const __half*)up, out, rows, cols, sum_norm, row_stride);
            break;
        case 2: // bf16
            silu_mul_q8a128_bf16<<<grid, block, 0, stream>>>((const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, out, rows, cols, sum_norm, row_stride);
            break;
    }
}

// =============================================================================
// FUSED MoE GATHER / WEIGHTED SCATTER-ADD
// =============================================================================
// Grid: (total_rows, ceil(hidden_dim / BLOCK_SIZE)) — 2D grid

void run_moe_gather(int32_t dtype, void* out, const void* xs,
                    const uint32_t* token_ids,
                    size_t total_rows, size_t hidden_dim, cudaStream_t stream) {
    if (total_rows == 0) return;
    dim3 grid(total_rows, grid_size(hidden_dim));
    switch (dtype) {
        case 0: // f32
            moe_gather_f32<<<grid, BLOCK_SIZE, 0, stream>>>((float*)out, (const float*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 1: // f16
            moe_gather_f16<<<grid, BLOCK_SIZE, 0, stream>>>((__half*)out, (const __half*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 2: // bf16
            moe_gather_bf16<<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16*)out, (const __nv_bfloat16*)xs, token_ids, total_rows, hidden_dim);
            break;
        case 3: { // q8a128 tile gather; hidden_dim = 128-element tiles per row
            // One warp per output tile, 8 warps a block, grid-strided.
            const size_t tiles = total_rows * hidden_dim;
            const unsigned blocks = (unsigned)((tiles + 7) / 8);
            moe_gather_q8a128_tiles<<<blocks > 0 ? blocks : 1, 256, 0, stream>>>(
                (uint8_t*)out, (const uint8_t*)xs, token_ids, total_rows, hidden_dim);
            break;
        }
    }
}

// Fused router. One thread per token (grid-stride). `logits` is [num_tokens, n_experts]
// in `dtype` (0=f32,1=f16,2=bf16); writes top-k expert indices (u32) and routing weights
// (f32), both [num_tokens, k] in descending-logit order. `norm_topk` selects renormalized
// top-k softmax (1) vs plain full-softmax weights (0).
// `row_stride` — the logits' row pitch in elements, at least `n_experts`.
void run_moe_route(int32_t dtype, const void* logits, uint32_t* out_idx, float* out_weights,
                   int num_tokens, int n_experts, int row_stride, int k, int norm_topk,
                   cudaStream_t stream) {
    if (num_tokens == 0 || row_stride < n_experts) return;
    // One warp per token; 256-thread blocks pack 8 warps (= 8 tokens) each.
    const int tpb = 256;
    const int warps_per_block = tpb / 32;
    const int blocks = (num_tokens + warps_per_block - 1) / warps_per_block;
    // The slot width is an instantiation, chosen by the expert count: the
    // 8-slot kernels serve up to 256, the 16-slot `_x512` variants up to 512.
    const bool wide = n_experts > 256;
    switch (dtype) {
        case 0: // f32
            if (wide) moe_route_f32_x512<<<blocks, tpb, 0, stream>>>((const float*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            else      moe_route_f32<<<blocks, tpb, 0, stream>>>((const float*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            break;
        case 1: // f16
            if (wide) moe_route_f16_x512<<<blocks, tpb, 0, stream>>>((const __half*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            else      moe_route_f16<<<blocks, tpb, 0, stream>>>((const __half*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            break;
        case 2: // bf16
            if (wide) moe_route_bf16_x512<<<blocks, tpb, 0, stream>>>((const __nv_bfloat16*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            else      moe_route_bf16<<<blocks, tpb, 0, stream>>>((const __nv_bfloat16*)logits, out_idx, out_weights, num_tokens, n_experts, row_stride, k, norm_topk);
            break;
    }
}

// Router look-ahead votes (`moe_scatter.cu`): one block per hop, a warp per
// token, `out` [hops][n_experts] u32 words of `(tokens routing it in their top
// k) << 16 | (tokens reaching it only in ranks k+1 … n)`, and `mass` [hops]
// [n_experts] f32 the softmax probability those picks carried. The hops' logits are
// `n_experts`-wide column runs from `first_col` of `row_stride`-wide f32 rows.
void run_moe_predict_votes(const float* logits, uint32_t* out, float* mass, int num_tokens,
                           int n_experts, int row_stride, int first_col, int hops, int k, int n,
                           cudaStream_t stream) {
    if (hops <= 0 || n_experts <= 0 || n_experts > 512) return;
    const int warps = num_tokens < 1 ? 1 : (num_tokens < 8 ? num_tokens : 8);
    moe_predict_votes_f32<<<hops, warps * 32, 0, stream>>>(
        logits, out, mass, num_tokens, n_experts, row_stride, first_col, k, n);
}

// Deterministic MoE scatter: sequential per-token reduce, no atomicAdd.
// perm[i] maps token-major index i to the expert-major row in down_out.
// grid: (num_tokens, ceil(hidden_dim / BLOCK_SIZE))
void run_deterministic_scatter(int32_t dtype, void* ys, const void* down_out,
                               const uint32_t* perm,
                               const float* weights_flat,
                               const uint32_t* reordered_weight_ids,
                               const int* token_starts,
                               int num_tokens, int hidden, cudaStream_t stream) {
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, grid_size(hidden));
    switch (dtype) {
        case 0: // f32
            deterministic_scatter_f32<<<grid, BLOCK_SIZE, 0, stream>>>(
                (float*)ys, (const float*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
        case 1: // f16
            deterministic_scatter_f16<<<grid, BLOCK_SIZE, 0, stream>>>(
                (__half*)ys, (const __half*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
        case 2: // bf16
            deterministic_scatter_bf16<<<grid, BLOCK_SIZE, 0, stream>>>(
                (__nv_bfloat16*)ys, (const __nv_bfloat16*)down_out,
                perm, weights_flat, reordered_weight_ids, token_starts, num_tokens, hidden);
            break;
    }
}
