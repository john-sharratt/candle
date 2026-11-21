// The block-size constants, the forward declarations of the quantize,
// dequantize and mul-mat kernels, and the integer helpers.
//
// Holds the `CUDA_*_BLOCK_SIZE` and `MATRIX_ROW_PADDING` constants, the forward
// declarations of the `quantize_*`, `dequantize_block_*` and
// `dequantize_mul_mat_vec_*` kernels the dispatchers call, the
// `DECLARE_MUL_MAT_VEC_Q8_1` expansion, the `mul_mat_*` kernels with their
// batch-narrow `_x16` forms, and the `ceil_div` and `pad` helpers.

// =============================================================================
// Constants
// =============================================================================

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define MATRIX_ROW_PADDING 512
// Note: WARP_SIZE is already defined in blocks.cuh
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define GGML_CUDA_MMV_Y 2

// =============================================================================
// Forward declarations of quantized.cu kernels
// =============================================================================

// Legacy quantize kernel (padded interface)
extern "C" __global__ void quantize_q8_1(const float*, void*, const int, const int);

// =============================================================================
// Forward declarations of quantize/*.cuh kernels
// =============================================================================

// Note: Block types are now defined via blocks.cuh include above.
// Keep forward declarations for kernels that use these types.

// Quantize tensor kernels (from quantize/quantize.cuh)
extern "C" __global__ void quantize_tensor_q4_0(const float*, block_q4_0*, int);
extern "C" __global__ void quantize_tensor_q4_1(const float*, block_q4_1*, int);
extern "C" __global__ void quantize_tensor_q5_0(const float*, block_q5_0*, int);
extern "C" __global__ void quantize_tensor_q5_1(const float*, block_q5_1*, int);
extern "C" __global__ void quantize_tensor_q8_0(const float*, block_q8_0*, int);
extern "C" __global__ void quantize_tensor_q8_1(const float*, block_q8_1*, int);
extern "C" __global__ void quantize_tensor_q4_ks(const float*, block_q4_ks*, int);
extern "C" __global__ void quantize_tensor_q8_ks(const float*, block_q8_ks*, int);
extern "C" __global__ void quantize_tensor_q2_0(const float*, block_q2_0*, int);
extern "C" __global__ void quantize_tensor_q3_0(const float*, block_q3_0*, int);
extern "C" __global__ void quantize_tensor_q2_K(const float*, block_q2_K*, int);
extern "C" __global__ void quantize_tensor_q3_K(const float*, block_q3_K*, int);
extern "C" __global__ void quantize_tensor_q4_K(const float*, block_q4_K*, int);
extern "C" __global__ void quantize_tensor_q5_K(const float*, block_q5_K*, int);
extern "C" __global__ void quantize_tensor_q6_K(const float*, block_q6_K*, int);
extern "C" __global__ void quantize_tensor_q8_K(const float*, block_q8_K*, int);
// AWQ quantize kernels (from quantize_kernels.cu with 80-byte padded structs)
// Note: We pass void* since the 80-byte padded structs aren't defined here
extern "C" __global__ void quantize_tensor_q_awq(const float*, void*, int);
extern "C" __global__ void quantize_tensor_q_awq_g64(const float*, void*, int);

// Dequantize block kernels (K-quants - no k parameter)
extern "C" __global__ void dequantize_block_q2_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q2_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q2_K_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q3_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q3_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q3_K_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q4_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q4_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q4_K_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q5_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q5_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q5_K_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q6_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q6_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q6_K_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q8_K_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q8_K_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q8_K_bf16(const void*, __nv_bfloat16*);

// Dequantize block kernels (basic quants - with k parameter)
extern "C" __global__ void dequantize_block_q4_0_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q4_0_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q4_0_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q4_1_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q4_1_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q4_1_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q5_0_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q5_0_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q5_0_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q5_1_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q5_1_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q5_1_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q8_0_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q8_0_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q8_0_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q8_1_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q8_1_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q8_1_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_mxfp4_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_mxfp4_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_mxfp4_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q4_ks_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q4_ks_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q4_ks_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q8_ks_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q8_ks_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q8_ks_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q2_0_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q2_0_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q2_0_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_q3_0_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_q3_0_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_q3_0_bf16(const void*, __nv_bfloat16*, const int);
extern "C" __global__ void dequantize_block_r16_f32(const void*, float*, const int);
extern "C" __global__ void dequantize_block_r16_f16(const void*, __half*, const int);
extern "C" __global__ void dequantize_block_r16_bf16(const void*, __nv_bfloat16*, const int);

// Dequantize block kernels (K/128 AWQ types - no k parameter)
extern "C" __global__ void dequantize_block_q_awq_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q_awq_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q_awq_bf16(const void*, __nv_bfloat16*);
extern "C" __global__ void dequantize_block_q_awq_g64_f32(const void*, float*);
extern "C" __global__ void dequantize_block_q_awq_g64_f16(const void*, __half*);
extern "C" __global__ void dequantize_block_q_awq_g64_bf16(const void*, __nv_bfloat16*);

// Dequantize mul mat vec kernels
extern "C" __global__ void dequantize_mul_mat_vec_q4_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q4_1_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_1_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q8_0_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q8_1_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q2_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q3_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q4_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q5_k(const void*, const float*, float*, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q6_k(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q8_k(const void*, const float*, float*, const int, const int);
// No AWQ: it is a KV arena format here, not a matmul weight format. Its
// dequantize/quantize kernels below serve the cache; its matmul kernels are
// gone, and `QCudaStorage::fwd` refuses an AWQ weight before reaching any of
// these switches. See that guard for why the layout never worked.

// Mul mat vec via Q8_1 kernels (batch sizes 1-8)
#define DECLARE_MUL_MAT_VEC_Q8_1(qtype) \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda1(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda2(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda3(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda4(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda5(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda6(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda7(const void*, const void*, float*, const int, const int, const int, const int); \
    extern "C" __global__ void mul_mat_vec_##qtype##_q8_1_cuda8(const void*, const void*, float*, const int, const int, const int, const int);

DECLARE_MUL_MAT_VEC_Q8_1(q4_0)
DECLARE_MUL_MAT_VEC_Q8_1(q4_1)
DECLARE_MUL_MAT_VEC_Q8_1(q5_0)
DECLARE_MUL_MAT_VEC_Q8_1(q5_1)
DECLARE_MUL_MAT_VEC_Q8_1(q8_0)
DECLARE_MUL_MAT_VEC_Q8_1(q8_1)
DECLARE_MUL_MAT_VEC_Q8_1(q2_K)
DECLARE_MUL_MAT_VEC_Q8_1(q3_K)
DECLARE_MUL_MAT_VEC_Q8_1(q4_K)
DECLARE_MUL_MAT_VEC_Q8_1(q5_K)
DECLARE_MUL_MAT_VEC_Q8_1(q6_K)
DECLARE_MUL_MAT_VEC_Q8_1(q8_K)

#undef DECLARE_MUL_MAT_VEC_Q8_1

// Mul mat kernels (tensor cores)
extern "C" __global__ void mul_mat_q4_0(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q4_1(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_0(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_1(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_0(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_1(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q2_K(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q3_K(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q4_K(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_K(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q6_K(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_K(const void*, const void*, float*, const int, const int, const int, const int, const int);

// The batch-narrow (mmq_x = 16) instantiations of the same kernels, selected
// when the whole batch fits one narrow tile — see run_mul_mat.
extern "C" __global__ void mul_mat_q4_0_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q4_1_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_0_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_1_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_0_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_1_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q2_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q3_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q4_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q5_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q6_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q8_K_x16(const void*, const void*, float*, const int, const int, const int, const int, const int);

// =============================================================================
// Helper functions
// =============================================================================

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int pad(int a, int b) {
    return ceil_div(a, b) * b;
}
