// =============================================================================
// DEQUANTIZE KERNEL INSTANTIATION AND DISPATCHER
// =============================================================================
// Dequantizes repacked (GEMX) quantized tensors using the actual matmul
// loaders. This tests the exact same element mapping as the matmul kernel.
// Note: K/128 blocks have embedded scales - no external scales parameter.
// =============================================================================

#include "dequantize.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// KERNEL WRAPPERS
// K/128 blocks have embedded scales - no external scales parameter needed.
// =============================================================================

// Q4_0 (qtype=0)
extern "C" __global__ void dequantize_q4_0(
    const block_c_q4_0* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q4_0_impl(x, out, nrows, ncols);
}

// Q4_1 (qtype=1)
extern "C" __global__ void dequantize_q4_1(
    const block_c_q4_1* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q4_1_impl(x, out, nrows, ncols);
}

// Q5_0 (qtype=2)
extern "C" __global__ void dequantize_q5_0(
    const block_c_q5_0* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q5_0_impl(x, out, nrows, ncols);
}

// Q5_1 (qtype=3)
extern "C" __global__ void dequantize_q5_1(
    const block_c_q5_1* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q5_1_impl(x, out, nrows, ncols);
}

// Q8_0 (qtype=4)
extern "C" __global__ void dequantize_q8_0(
    const block_c_q8_0* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q8_0_impl(x, out, nrows, ncols);
}

// Q2_K (qtype=5)
extern "C" __global__ void dequantize_q2_K(
    const block_c_q2_K* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q2_K_impl(x, out, nrows, ncols);
}

// Q3_K (qtype=6)
extern "C" __global__ void dequantize_q3_K(
    const block_c_q3_K* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q3_K_impl(x, out, nrows, ncols);
}

// Q4_K (qtype=7)
extern "C" __global__ void dequantize_q4_K(
    const block_c_q4_K* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q4_K_impl(x, out, nrows, ncols);
}

// Q5_K (qtype=8)
extern "C" __global__ void dequantize_q5_K(
    const block_c_q5_K* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q5_K_impl(x, out, nrows, ncols);
}

// Q6_K (qtype=9)
extern "C" __global__ void dequantize_q6_K(
    const block_c_q6_K* __restrict__ x,
    float* __restrict__ out,
    int nrows,
    int ncols
) {
    dequantize_q6_K_impl(x, out, nrows, ncols);
}

// =============================================================================
// DISPATCHER
// =============================================================================

/// Dequantize repacked quantized tensor to float32 using the matmul loader.
///
/// This uses the exact same element mapping as the matmul kernel's loader,
/// allowing direct debugging of the loader's element indexing.
///
/// Parameters:
/// - x: Repacked blocks with embedded scales (device pointer)
/// - out: Output float32 buffer (device pointer, nrows × ncols)
/// - nrows: Number of rows
/// - ncols: Number of columns
/// - qtype: Quantization type (0-9)
///
/// Note: K/128 blocks have embedded scales - no external scales parameter.
///
/// Returns: 0 on success, -1 on error
extern "C" int32_t run_dequantize(
    const void* x,
    void* out,
    int32_t nrows,
    int32_t ncols,
    int32_t qtype
) {
    // Get QK and threads per block based on qtype
    // K/128 FORMAT: QK=128 (8 K-tiles × 16 elements), 16 threads per block
    int QK = 128;
    int threads_per_quant_block;
    
    switch (qtype) {
        case 0: // Q4_0: 16 threads per K/128 block
        case 1: // Q4_1
        case 2: // Q5_0
        case 3: // Q5_1
            threads_per_quant_block = 16;  // qi=32, vdr=2 → 16 threads
            break;
        case 4: // Q8_0
            threads_per_quant_block = 16;  // qi=16, vdr=1 → 16 threads
            break;
        case 5: // Q2_K
            threads_per_quant_block = 64;  // qi=64, vdr=1
            break;
        case 6: // Q3_K
            threads_per_quant_block = 16;  // qi=16, vdr=1
            break;
        case 7: // Q4_K
        case 8: // Q5_K
        case 9: // Q6_K
            threads_per_quant_block = 16;  // qi=32, vdr=2
            break;
        default:
            return -1;
    }
    
    if (ncols % QK != 0) {
        return -1;
    }
    
    const int blocks_per_row = ncols / QK;
    const int total_blocks = nrows * blocks_per_row;
    const int total_threads = total_blocks * threads_per_quant_block;
    
    constexpr int THREADS_PER_CUDA_BLOCK = 256;
    const int num_cuda_blocks = (total_threads + THREADS_PER_CUDA_BLOCK - 1) / THREADS_PER_CUDA_BLOCK;
    
    switch (qtype) {
        case 0:
            dequantize_q4_0<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q4_0*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 1:
            dequantize_q4_1<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q4_1*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 2:
            dequantize_q5_0<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q5_0*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 3:
            dequantize_q5_1<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q5_1*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 4:
            dequantize_q8_0<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q8_0*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 5:
            dequantize_q2_K<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q2_K*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 6:
            dequantize_q3_K<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q3_K*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 7:
            dequantize_q4_K<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q4_K*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 8:
            dequantize_q5_K<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q5_K*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        case 9:
            dequantize_q6_K<<<num_cuda_blocks, THREADS_PER_CUDA_BLOCK>>>(
                reinterpret_cast<const block_c_q6_K*>(x),
                reinterpret_cast<float*>(out),
                nrows, ncols
            );
            break;
        default:
            return -1;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1;
    }
    
    return 0;
}

/// Get the output size (in floats) for dequantizing a tensor.
extern "C" int64_t get_dequantize_output_size(int32_t nrows, int32_t ncols) {
    return (int64_t)nrows * ncols;
}
