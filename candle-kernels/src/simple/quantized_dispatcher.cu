// =============================================================================
// QUANTIZED OPERATIONS DISPATCHER
// =============================================================================
// Provides extern "C" entry points that dispatch to the appropriate
// quantized kernels based on dtype parameters.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Include block type definitions for fused transpose+quantize kernels
#include "../blocks.cuh"

// QTYPE_* symbolic constants (GgmlDType-aligned numbering, source of truth)
#include "../quantized/block_compact.cuh"

// Include batched fused transpose+quantize kernels
#include "../quantize/transpose_batch.cuh"

// Include palette4 format conversion kernel (new replacement, coexists with above)
#include "../quantize/palette4_convert.cuh"

// Include adaptive per-block format selection kernel
#include "../quantize/select_kv_format.cuh"

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
extern "C" __global__ void dequantize_mul_mat_vec_q_awq_cuda(const void*, const float*, float*, const int, const int);
extern "C" __global__ void dequantize_mul_mat_vec_q_awq_g64_cuda(const void*, const float*, float*, const int, const int);

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
DECLARE_MUL_MAT_VEC_Q8_1(q_awq)
DECLARE_MUL_MAT_VEC_Q8_1(q_awq_g64)

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
extern "C" __global__ void mul_mat_q_awq(const void*, const void*, float*, const int, const int, const int, const int, const int);
extern "C" __global__ void mul_mat_q_awq_g64(const void*, const void*, float*, const int, const int, const int, const int, const int);

// =============================================================================
// Helper functions
// =============================================================================

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__host__ __device__ inline int pad(int a, int b) {
    return ceil_div(a, b) * b;
}

// =============================================================================
// DISPATCHER: quantize_q8_1 (legacy interface)
// =============================================================================

extern "C" void run_quantize_q8_1(
    const float* src,
    void* dst,
    int32_t elem_count,
    int32_t ky
) {
    int kx = elem_count;
    int kx_padded = pad(kx, MATRIX_ROW_PADDING);
    int num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    
    dim3 grid(num_blocks, ky, 1);
    dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    
    quantize_q8_1<<<grid, block>>>(src, dst, kx, kx_padded);
}

// =============================================================================
// DISPATCHER: quantize_block (unified interface)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
//
// elem_count: total number of f32 elements to quantize
// The function calculates number of quantized blocks based on qtype's block size

extern "C" void run_quantize_block(
    const float* src,
    void* dst,
    int32_t elem_count,
    int32_t qtype
) {
    // Calculate number of quantized blocks based on format
    int num_blocks;
    int block_size;  // elements per quantized block
    int threads_per_block = 32;  // warp size - our kernels are warp-cooperative
    bool is_k_quant = false;  // K-quants use __shared__ memory, need 1 warp per block

    switch (qtype) {
        // Standard 32-element formats
        case QTYPE_Q4_0:
        case QTYPE_Q4_1:
        case QTYPE_Q5_0:
        case QTYPE_Q5_1:
        case QTYPE_Q8_0:
        case QTYPE_Q8_1:
        case QTYPE_Q4_KS:
        case QTYPE_Q8_KS:
        case QTYPE_Q2_0:
        case QTYPE_Q3_0:
            block_size = 32;
            break;
        // K-quant 256-element formats
        case QTYPE_Q2_K:
        case QTYPE_Q3_K:
        case QTYPE_Q4_K:
        case QTYPE_Q5_K:
        case QTYPE_Q6_K:
        case QTYPE_Q8_K:
            block_size = 256;
            is_k_quant = true;
            break;
        // AWQ 128-element formats (both use 128 elements per block)
        case QTYPE_QAWQ:
        case QTYPE_QAWQ_G64:
            block_size = 128;
            break;
        default:
            return;  // Invalid qtype
    }

    num_blocks = ceil_div(elem_count, block_size);

    // K-quants use __shared__ memory that's shared across all warps in a thread block.
    // Each warp needs exclusive access to shared memory, so we must launch with 1 warp per block.
    // Standard quants don't use shared memory, so they can use multiple warps per block.
    dim3 grid, block_dim;
    if (is_k_quant) {
        // K-quants: 1 warp (32 threads) per thread block - each block gets exclusive shared memory
        grid = dim3(num_blocks, 1, 1);
        block_dim = dim3(threads_per_block, 1, 1);  // 32 threads = 1 warp
    } else {
        // Standard quants: 4 warps per thread block for better occupancy
        int blocks_per_grid = ceil_div(num_blocks, 4);
        grid = dim3(blocks_per_grid, 1, 1);
        block_dim = dim3(threads_per_block * 4, 1, 1);  // 128 threads = 4 warps
    }

    switch (qtype) {
        case QTYPE_Q4_0:
            quantize_tensor_q4_0<<<grid, block_dim>>>(src, (block_q4_0*)dst, num_blocks);
            break;
        case QTYPE_Q4_1:
            quantize_tensor_q4_1<<<grid, block_dim>>>(src, (block_q4_1*)dst, num_blocks);
            break;
        case QTYPE_Q5_0:
            quantize_tensor_q5_0<<<grid, block_dim>>>(src, (block_q5_0*)dst, num_blocks);
            break;
        case QTYPE_Q5_1:
            quantize_tensor_q5_1<<<grid, block_dim>>>(src, (block_q5_1*)dst, num_blocks);
            break;
        case QTYPE_Q8_0:
            quantize_tensor_q8_0<<<grid, block_dim>>>(src, (block_q8_0*)dst, num_blocks);
            break;
        case QTYPE_Q2_K:
            quantize_tensor_q2_K<<<grid, block_dim>>>(src, (block_q2_K*)dst, num_blocks);
            break;
        case QTYPE_Q3_K:
            quantize_tensor_q3_K<<<grid, block_dim>>>(src, (block_q3_K*)dst, num_blocks);
            break;
        case QTYPE_Q4_K:
            quantize_tensor_q4_K<<<grid, block_dim>>>(src, (block_q4_K*)dst, num_blocks);
            break;
        case QTYPE_Q5_K:
            quantize_tensor_q5_K<<<grid, block_dim>>>(src, (block_q5_K*)dst, num_blocks);
            break;
        case QTYPE_Q6_K:
            quantize_tensor_q6_K<<<grid, block_dim>>>(src, (block_q6_K*)dst, num_blocks);
            break;
        case QTYPE_Q8_1:
            quantize_tensor_q8_1<<<grid, block_dim>>>(src, (block_q8_1*)dst, num_blocks);
            break;
        case QTYPE_Q4_KS:
            quantize_tensor_q4_ks<<<grid, block_dim>>>(src, (block_q4_ks*)dst, num_blocks);
            break;
        case QTYPE_Q8_KS:
            quantize_tensor_q8_ks<<<grid, block_dim>>>(src, (block_q8_ks*)dst, num_blocks);
            break;
        case QTYPE_Q2_0:
            quantize_tensor_q2_0<<<grid, block_dim>>>(src, (block_q2_0*)dst, num_blocks);
            break;
        case QTYPE_Q3_0:
            quantize_tensor_q3_0<<<grid, block_dim>>>(src, (block_q3_0*)dst, num_blocks);
            break;
        case QTYPE_Q8_K:
            quantize_tensor_q8_K<<<grid, block_dim>>>(src, (block_q8_K*)dst, num_blocks);
            break;
        case QTYPE_QAWQ:
            quantize_tensor_q_awq<<<grid, block_dim>>>(src, dst, num_blocks);
            break;
        case QTYPE_QAWQ_G64:
            quantize_tensor_q_awq_g64<<<grid, block_dim>>>(src, dst, num_blocks);
            break;
    }
}

// =============================================================================
// DISPATCHER: dequantize_block
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
// out_dtype: 0=F32, 1=F16, 2=BF16

extern "C" void run_dequantize_block(
    const void* src,
    void* dst,
    int32_t elem_count,
    int32_t qtype,
    int32_t out_dtype
) {
    // Determine grid/block dimensions based on quant type
    int block_dim;
    int num_blocks;
    int nb32 = 0;

    bool is_k128_type = (qtype == QTYPE_QAWQ || qtype == QTYPE_QAWQ_G64);  // K/128 AWQ native types
    bool is_k_quant =
        (qtype == QTYPE_Q2_K || qtype == QTYPE_Q3_K || qtype == QTYPE_Q4_K ||
         qtype == QTYPE_Q5_K || qtype == QTYPE_Q6_K || qtype == QTYPE_Q8_K);

    if (is_k128_type) {
        // K/128 types: 128 elements per block, 32 threads
        block_dim = 32;
        num_blocks = ceil_div(elem_count, 128);
    } else if (is_k_quant) {
        // GGML K-quants: 256 elements per block
        block_dim = (qtype == QTYPE_Q4_K) ? 32 : 64;
        num_blocks = ceil_div(elem_count, 256);
    } else {
        // Basic quants: 32 elements per block
        if (qtype == QTYPE_Q5_0 || qtype == QTYPE_Q5_1) {
            block_dim = CUDA_DEQUANTIZE_BLOCK_SIZE;
            num_blocks = ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE);
            nb32 = elem_count;
        } else {
            block_dim = 32;
            num_blocks = ceil_div(elem_count, 256);
            nb32 = elem_count / 32;
        }
    }

    dim3 grid(num_blocks, 1, 1);
    dim3 block(block_dim, 1, 1);

    // Dispatch based on qtype and out_dtype
    switch (qtype) {
        // Basic quants (with k parameter)
        case QTYPE_Q4_0:
            if (out_dtype == 0) dequantize_block_q4_0_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_0_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_0_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q4_1:
            if (out_dtype == 0) dequantize_block_q4_1_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_1_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_1_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q5_0:
            if (out_dtype == 0) dequantize_block_q5_0_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q5_0_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q5_0_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q5_1:
            if (out_dtype == 0) dequantize_block_q5_1_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q5_1_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q5_1_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q8_0:
            if (out_dtype == 0) dequantize_block_q8_0_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_0_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_0_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;

        // K-quants (no k parameter)
        case QTYPE_Q2_K:
            if (out_dtype == 0) dequantize_block_q2_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q2_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q2_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q3_K:
            if (out_dtype == 0) dequantize_block_q3_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q3_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q3_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q4_K:
            if (out_dtype == 0) dequantize_block_q4_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q4_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q4_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q5_K:
            if (out_dtype == 0) dequantize_block_q5_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q5_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q5_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q6_K:
            if (out_dtype == 0) dequantize_block_q6_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q6_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q6_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_Q8_1:
            if (out_dtype == 0) dequantize_block_q8_1_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_1_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_1_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q4_KS:
            if (out_dtype == 0) dequantize_block_q4_ks_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q4_ks_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q4_ks_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q8_KS:
            if (out_dtype == 0) dequantize_block_q8_ks_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q8_ks_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q8_ks_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q2_0:
            if (out_dtype == 0) dequantize_block_q2_0_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q2_0_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q2_0_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_Q3_0:
            if (out_dtype == 0) dequantize_block_q3_0_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_q3_0_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_q3_0_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;
        case QTYPE_R16: // R16 (extract K from block_r16::d[])
            if (out_dtype == 0) dequantize_block_r16_f32<<<grid, block>>>(src, (float*)dst, nb32);
            else if (out_dtype == 1) dequantize_block_r16_f16<<<grid, block>>>(src, (__half*)dst, nb32);
            else dequantize_block_r16_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst, nb32);
            break;

        // Q8_K (K-quant 8-bit, 256 elements per block)
        case QTYPE_Q8_K:
            if (out_dtype == 0) dequantize_block_q8_K_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q8_K_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q8_K_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;

        // K/128 AWQ types (no k parameter, 128 elements per block)
        case QTYPE_QAWQ:
            if (out_dtype == 0) dequantize_block_q_awq_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q_awq_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q_awq_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
        case QTYPE_QAWQ_G64:
            if (out_dtype == 0) dequantize_block_q_awq_g64_f32<<<grid, block>>>(src, (float*)dst);
            else if (out_dtype == 1) dequantize_block_q_awq_g64_f16<<<grid, block>>>(src, (__half*)dst);
            else dequantize_block_q_awq_g64_bf16<<<grid, block>>>(src, (__nv_bfloat16*)dst);
            break;
    }
}

// =============================================================================
// DISPATCHER: dequantize_mul_mat_vec
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).

extern "C" void run_dequantize_mul_mat_vec(
    const void* vx,
    const float* y,
    float* dst,
    int32_t ncols,
    int32_t nrows,
    int32_t qtype
) {
    int block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);

    dim3 grid(block_num_y, 1, 1);
    dim3 block(WARP_SIZE, GGML_CUDA_MMV_Y, 1);

    switch (qtype) {
        case QTYPE_Q4_0: dequantize_mul_mat_vec_q4_0_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q4_1: dequantize_mul_mat_vec_q4_1_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_0: dequantize_mul_mat_vec_q5_0_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_1: dequantize_mul_mat_vec_q5_1_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_0: dequantize_mul_mat_vec_q8_0_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q2_K: dequantize_mul_mat_vec_q2_k<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q3_K: dequantize_mul_mat_vec_q3_k<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q4_K: dequantize_mul_mat_vec_q4_k<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q5_K: dequantize_mul_mat_vec_q5_k<<<grid, block>>>(vx, y, dst, ncols); break;  // q5_k has different signature
        case QTYPE_Q6_K: dequantize_mul_mat_vec_q6_k<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_1: dequantize_mul_mat_vec_q8_1_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_Q8_K: dequantize_mul_mat_vec_q8_k<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_QAWQ: dequantize_mul_mat_vec_q_awq_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
        case QTYPE_QAWQ_G64: dequantize_mul_mat_vec_q_awq_g64_cuda<<<grid, block>>>(vx, y, dst, ncols, nrows); break;
    }
}

// =============================================================================
// DISPATCHER: mul_mat_vec_q8_1 (batched)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).
// b_size: 1-8

extern "C" void run_mul_mat_vec_q8_1(
    const void* vx,
    const void* vy,
    float* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t b_size,
    int32_t qtype
) {
    // Compute grid/block dimensions based on batch size
    int nblocks, nwarps;
    switch (b_size) {
        case 1:
            nblocks = nrows_x;
            nwarps = 4;
            break;
        case 2:
        case 3:
        case 4:
            nblocks = ceil_div(nrows_x, 2);
            nwarps = 4;
            break;
        default:  // 5-8
            nblocks = ceil_div(nrows_x, 2);
            nwarps = 2;
            break;
    }
    
    dim3 grid(nblocks, 1, 1);
    dim3 block(WARP_SIZE, nwarps, 1);
    
    // Macro to dispatch based on batch size
    #define DISPATCH_BSIZE(qname) \
        switch (b_size) { \
            case 1: mul_mat_vec_##qname##_q8_1_cuda1<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 2: mul_mat_vec_##qname##_q8_1_cuda2<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 3: mul_mat_vec_##qname##_q8_1_cuda3<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 4: mul_mat_vec_##qname##_q8_1_cuda4<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 5: mul_mat_vec_##qname##_q8_1_cuda5<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 6: mul_mat_vec_##qname##_q8_1_cuda6<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 7: mul_mat_vec_##qname##_q8_1_cuda7<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
            case 8: mul_mat_vec_##qname##_q8_1_cuda8<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); break; \
        }
    
    switch (qtype) {
        case QTYPE_Q4_0: DISPATCH_BSIZE(q4_0); break;
        case QTYPE_Q4_1: DISPATCH_BSIZE(q4_1); break;
        case QTYPE_Q5_0: DISPATCH_BSIZE(q5_0); break;
        case QTYPE_Q5_1: DISPATCH_BSIZE(q5_1); break;
        case QTYPE_Q8_0: DISPATCH_BSIZE(q8_0); break;
        case QTYPE_Q2_K: DISPATCH_BSIZE(q2_K); break;
        case QTYPE_Q3_K: DISPATCH_BSIZE(q3_K); break;
        case QTYPE_Q4_K: DISPATCH_BSIZE(q4_K); break;
        case QTYPE_Q5_K: DISPATCH_BSIZE(q5_K); break;
        case QTYPE_Q6_K: DISPATCH_BSIZE(q6_K); break;
        case QTYPE_Q8_1: DISPATCH_BSIZE(q8_1); break;
        case QTYPE_Q8_K: DISPATCH_BSIZE(q8_K); break;
        case QTYPE_QAWQ: DISPATCH_BSIZE(q_awq); break;
        case QTYPE_QAWQ_G64: DISPATCH_BSIZE(q_awq_g64); break;
    }

    #undef DISPATCH_BSIZE
}

// =============================================================================
// DISPATCHER: mul_mat (tensor core / MMQ)
// =============================================================================
// qtype: GgmlDType-aligned integer (see QTYPE_* in quantized/block_compact.cuh).

extern "C" void run_mul_mat(
    const void* vx,
    const void* vy,
    float* dst,
    int32_t ncols_x,
    int32_t nrows_x,
    int32_t ncols_y,
    int32_t nrows_y,
    int32_t nrows_dst,
    int32_t qtype
) {
    // Get MMQ tile sizes based on quant type
    int mmq_x, mmq_y;
    switch (qtype) {
        case QTYPE_Q4_0:
        case QTYPE_Q4_1:
        case QTYPE_Q2_K:
        case QTYPE_Q4_K:
        case QTYPE_Q5_K:
            mmq_x = 64; mmq_y = 128;
            break;
        case QTYPE_Q5_0:
        case QTYPE_Q5_1:
        case QTYPE_Q8_0:
            mmq_x = 128; mmq_y = 64;
            break;
        case QTYPE_Q3_K:
            mmq_x = 128; mmq_y = 128;
            break;
        case QTYPE_Q6_K:
        case QTYPE_Q8_K:
            mmq_x = 64; mmq_y = 64;
            break;
        default:
            mmq_x = 64; mmq_y = 128;
            break;
    }

    dim3 grid(ceil_div(nrows_x, mmq_y), ceil_div(ncols_y, mmq_x), 1);
    dim3 block(WARP_SIZE, 4, 1);

    switch (qtype) {
        case QTYPE_Q4_0: mul_mat_q4_0<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q4_1: mul_mat_q4_1<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q5_0: mul_mat_q5_0<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q5_1: mul_mat_q5_1<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q8_0: mul_mat_q8_0<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q2_K: mul_mat_q2_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q3_K: mul_mat_q3_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q4_K: mul_mat_q4_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q5_K: mul_mat_q5_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q6_K: mul_mat_q6_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q8_1: mul_mat_q8_1<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_Q8_K: mul_mat_q8_K<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_QAWQ: mul_mat_q_awq<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
        case QTYPE_QAWQ_G64: mul_mat_q_awq_g64<<<grid, block>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst); break;
    }
}

// =============================================================================
// Q0_V DEQUANTIZE TEST ENTRYPOINT
// =============================================================================
// Wraps the production GPU decoder (BlockConverter<block_q0_v, T>::load) in a
// standalone kernel so Rust unit tests can exercise the exact same decode path
// used by attention/prefill kernels. One warp per block; lane t emits dst[t].

extern "C" __global__ void dequantize_block_q0_v_f32_kernel(
    const block_q0_v* __restrict__ src,
    float* __restrict__ dst,
    int num_blocks,
    float scale)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (blk >= num_blocks) return;
    BlockConverter<block_q0_v, float>::load(
        dst + blk * QK_Q0_V, src + blk, lane, scale);
}

extern "C" void run_dequantize_block_q0_v_f32(
    const void* src, void* dst, int num_blocks, float scale)
{
    if (num_blocks <= 0) return;
    const int threads = 32;  // one warp per block
    const int blocks = num_blocks;
    dequantize_block_q0_v_f32_kernel<<<blocks, threads>>>(
        (const block_q0_v*)src, (float*)dst, num_blocks, scale);
}

// =============================================================================
// Q0_V ROUND-TRIP TEST ENTRYPOINTS  (K-side and V-side)
// =============================================================================
// These wrap the *actual* production quantize and dequantize paths used by
// the format-selection kernel. Used by the offline modelling/diagnostic
// test to measure round-trip error per block under the same code paths the
// kernel uses, isolating whether observed Q0_V usage gaps are due to the
// quant kernels themselves or the selection wiring.
//
// Layout:
//   - One warp per 32-element block (lane = element index)
//   - Encoder: warp-cooperative `quantize_block_q0_v_core<IS_K>`
//   - Decoder: warp-cooperative load via `q0_v_load_element_f32<IS_K>`
//   - Pre-scale: caller passes input pre-scaled by `outer` (i.e.
//     `src * outer`), encoder writes the resulting block.
//   - Post-scale: decoder divides by `outer` to recover original units;
//     for our metric work we pass `outer = 1.0` so the round-trip is in
//     the encoder's natural normalised space.

template <bool IS_K>
__device__ __forceinline__ void roundtrip_kernel_body(
    const float* __restrict__ src, float* __restrict__ recon,
    int blk, int num_blocks, int lane, float outer)
{
    if (blk >= num_blocks) return;
    const float xi = src[blk * QK_Q0_V + lane] * outer;
    block_q0_v packed;
    quantize_block_q0_v_core<IS_K>(xi, &packed);
    // All 32 lanes share the just-encoded block; broadcast lane 0's bytes
    // to every lane (warp-uniform after the encoder's lane-0 store).
    const unsigned lo_bits = __shfl_sync(0xffffffff, (unsigned)packed.lo, 0, 32);
    const unsigned hi_bits = __shfl_sync(0xffffffff, (unsigned)packed.hi, 0, 32);
    block_q0_v shared_packed;
    shared_packed.lo = (uint8_t)lo_bits;
    shared_packed.hi = (uint8_t)hi_bits;
    recon[blk * QK_Q0_V + lane] =
        q0_v_load_element_f32<IS_K>(&shared_packed, lane, outer);
}

extern "C" __global__ void roundtrip_q0_v_k_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    roundtrip_kernel_body<true>(src, recon, blk, num_blocks, lane, outer);
}

extern "C" __global__ void roundtrip_q0_v_v_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    roundtrip_kernel_body<false>(src, recon, blk, num_blocks, lane, outer);
}

extern "C" void run_roundtrip_q0_v(
    const void* src, void* recon, int num_blocks, float outer, int is_k)
{
    if (num_blocks <= 0) return;
    const int threads = 32;
    const int blocks = num_blocks;
    if (is_k) {
        roundtrip_q0_v_k_kernel<<<blocks, threads>>>(
            (const float*)src, (float*)recon, num_blocks, outer);
    } else {
        roundtrip_q0_v_v_kernel<<<blocks, threads>>>(
            (const float*)src, (float*)recon, num_blocks, outer);
    }
}

// =============================================================================
// Q0_V RUNTIME-TABLE ROUND-TRIP — diagnostic / curve-selection entry point
// =============================================================================
// Same warp-cooperative encoder + decoder pair the production path uses, but
// the codebook tables (curve, scale, centroid, peak permutation, peak bin
// offsets) are supplied as device pointers at launch time instead of read
// from `__constant__` memory. Used by the iterative curve-selection
// diagnostic to swap the codebook between iterations without recompiling.
//
// Caller layout contract (all on the device):
//   curve_table_flat       : [256][32] i8                    (8192 B)
//   scale_table_bits       : [32]      uint16 (f16 bits)     (   64 B)
//   centroid_table_bits    : [32][8]   uint16 (f16 bits)     (  512 B)
//   peak_curve_indices     : [256]     uint8                 (  256 B)
//   peak_bin_offsets       : [33]      uint16                (   66 B)
//
// Behaviour matches the static path byte-for-byte when the runtime tables
// are populated with the same content as the static `_k`/`_v` arrays.

extern "C" __global__ void roundtrip_q0_v_runtime_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer,
    const int8_t*   __restrict__ curve_table_flat,
    const uint16_t* __restrict__ scale_table_bits,
    const uint16_t* __restrict__ centroid_table_bits_flat,
    const uint8_t*  __restrict__ peak_curve_indices,
    const uint16_t* __restrict__ peak_bin_offsets)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (blk >= num_blocks) return;

    q0_v_detail::Q0VTablesRuntime tbl{
        curve_table_flat,
        scale_table_bits,
        centroid_table_bits_flat,
        peak_curve_indices,
        peak_bin_offsets,
    };

    const float xi = src[blk * QK_Q0_V + lane] * outer;
    block_q0_v packed;
    quantize_block_q0_v_core_runtime(xi, &packed, tbl);

    // Broadcast the lane-0 packed bytes warp-wide before decode.
    const unsigned lo_bits = __shfl_sync(0xffffffff, (unsigned)packed.lo, 0, 32);
    const unsigned hi_bits = __shfl_sync(0xffffffff, (unsigned)packed.hi, 0, 32);
    block_q0_v shared_packed;
    shared_packed.lo = (uint8_t)lo_bits;
    shared_packed.hi = (uint8_t)hi_bits;

    const float decoded = q0_v_elem_runtime(&shared_packed, lane, tbl);
    recon[blk * QK_Q0_V + lane] = decoded / outer;
}

extern "C" void run_roundtrip_q0_v_runtime(
    const void* src,
    void*       recon,
    int         num_blocks,
    float       outer,
    const void* curve_table_flat,         // [256][32] i8
    const void* scale_table_bits,          // [32]      u16
    const void* centroid_table_bits_flat,  // [32][8]   u16
    const void* peak_curve_indices,        // [256]     u8
    const void* peak_bin_offsets)          // [33]      u16
{
    if (num_blocks <= 0) return;
    const int threads = 32;
    const int blocks = num_blocks;
    roundtrip_q0_v_runtime_kernel<<<blocks, threads>>>(
        (const float*)src, (float*)recon, num_blocks, outer,
        (const int8_t*)curve_table_flat,
        (const uint16_t*)scale_table_bits,
        (const uint16_t*)centroid_table_bits_flat,
        (const uint8_t*)peak_curve_indices,
        (const uint16_t*)peak_bin_offsets);
}
