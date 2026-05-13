// =============================================================================
// GEMX WEIGHT REPACKING KERNEL INSTANTIATION AND DISPATCHER
// =============================================================================
// Instantiates weight repacking kernels for all GGML quantization formats
// and provides a unified dispatcher entry point.
//
// The repacking copies from src buffer to dst buffer (NOT in-place) to avoid
// race conditions with CUDA's arbitrary block execution order.
// After repacking:
// - Scales are stored separately (via extract_scales kernel)
// - Quants are reordered for GEMX tensor core kernel (K/128 format)
// - Total tensor size is reduced (scale bytes removed)
// =============================================================================

#include "repack_gemx.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// KERNEL WRAPPERS (C ABI for FFI - names kept for backwards compatibility)
// =============================================================================

// Q4_0 repacking kernel
extern "C" __global__ void repack_gemx_q4_0(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q4_0_impl<QK4_0>(src, dst, nrows, ncols);
}

// Q4_1 repacking kernel
extern "C" __global__ void repack_gemx_q4_1(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q4_1_impl<QK4_1>(src, dst, nrows, ncols);
}

// Q5_0 repacking kernel
extern "C" __global__ void repack_gemx_q5_0(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q5_0_impl<QK5_0>(src, dst, nrows, ncols);
}

// Q5_1 repacking kernel
extern "C" __global__ void repack_gemx_q5_1(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q5_1_impl<QK5_1>(src, dst, nrows, ncols);
}

// Q8_0 repacking kernel
extern "C" __global__ void repack_gemx_q8_0(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q8_0_impl<QK8_0>(src, dst, nrows, ncols);
}

// Q8_1 repacking kernel
extern "C" __global__ void repack_gemx_q8_1(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q8_1_impl<QK8_1>(src, dst, nrows, ncols);
}

// Q2_K repacking kernel
extern "C" __global__ void repack_gemx_q2_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q2_K_impl(src, dst, nrows, ncols);
}

// Q3_K repacking kernel
extern "C" __global__ void repack_gemx_q3_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q3_K_impl(src, dst, nrows, ncols);
}

// Q4_K repacking kernel
extern "C" __global__ void repack_gemx_q4_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q4_K_impl(src, dst, nrows, ncols);
}

// Q5_K repacking kernel
extern "C" __global__ void repack_gemx_q5_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q5_K_impl(src, dst, nrows, ncols);
}

// Q6_K repacking kernel
extern "C" __global__ void repack_gemx_q6_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q6_K_impl(src, dst, nrows, ncols);
}

// Q8_K repacking kernel
extern "C" __global__ void repack_gemx_q8_K(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q8_K_impl(src, dst, nrows, ncols);
}

// =============================================================================
// AWQ KERNELS
// =============================================================================

// Q_AWQ repacking kernel (group size 128)
extern "C" __global__ void repack_gemx_q_awq(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q_awq_impl(src, dst, nrows, ncols);
}

// Q_AWQ_G64 repacking kernel (group size 64)
extern "C" __global__ void repack_gemx_q_awq_g64(
    const void* src,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q_awq_g64_impl(src, dst, nrows, ncols);
}

// Q_AWQ repacking from HuggingFace format (separate qweight/qzeros/scales)
extern "C" __global__ void repack_gemx_q_awq_hf(
    const int32_t* qweight,
    const int32_t* qzeros,
    const half* scales,
    void* dst,
    int nrows,
    int ncols
) {
    repack_q_awq_hf_impl(qweight, qzeros, scales, dst, nrows, ncols);
}

// =============================================================================
// KERNEL DISPATCH MACRO
// =============================================================================
// Uses triple-chevron syntax for proper CUDA kernel launch.
// Cannot use function pointer table for __global__ kernels reliably.

#define LAUNCH_REPACK_KERNEL(kernel_name, grid, block, src, dst, nrows, ncols) \
    kernel_name<<<grid, block>>>(src, dst, nrows, ncols)

// =============================================================================
// DISPATCHER (C ABI for FFI - names kept for backwards compatibility)
// =============================================================================

/// Repack quantized weights to GEMX format (K/128 with embedded scales).
///
/// This removes scale data from the weights (scales should be extracted
/// separately via extract_scales before calling this) and reorders the
/// quant bytes for optimal tensor core access patterns.
///
/// Parameters:
/// - src_data: Source weight tensor data (device pointer, read-only)
/// - dst_data: Destination buffer (device pointer, must be pre-allocated to output_size)
/// - nrows: Number of rows in tensor
/// - ncols: Number of columns in tensor
/// - qtype: Quantization type (0-13, see QType enum in block_compact.cuh)
///
/// Returns: 0 on success, -1 on error
///
/// IMPORTANT: The output is smaller than the input since scales are removed.
/// Use get_repacked_size_bytes() to determine the required dst_data size.
extern "C" int32_t run_repack_gemx(
    const void* src_data,
    void* dst_data,
    int32_t nrows,
    int32_t ncols,
    int32_t qtype
) {
    // Gate on whether this qtype has a GEMX output layout. This delegates to
    // the single source of truth in `qtype_output_block_size` and avoids the
    // old hardcoded `qtype > 13` range check that dated from the pre-reorder
    // GGML indexing and rejected every format past Q5_K.
    if (qtype_output_block_size(qtype) <= 0) {
        printf("run_repack_gemx: unsupported qtype %d\n", qtype);
        return -1;
    }

    const int input_elems = qtype_input_block_elems(qtype);

    // Grid / block dims vary by input block granularity:
    //   - 256-elem super-blocks (K-quants other than Q8_K): 1 CUDA block per super-block, 32 threads
    //   - 128-elem blocks (Q8_K, AWQ, AWQ_G64):             1 CUDA block per K/128 output block, 32 threads
    //   - 32-elem blocks (simple Q4/Q5/Q8_0/Q8_1):          tiled with 256 threads per CUDA block
    dim3 grid;
    dim3 block;
    if (input_elems == 256 && qtype != QTYPE_Q8_K) {
        // Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
        const int total_blocks = nrows * (ncols / 256);
        grid = dim3(total_blocks);
        block = dim3(32);
    } else if (input_elems == 128 || qtype == QTYPE_Q8_K) {
        // Q8_K, QAWQ, QAWQ_G64
        const int total_blocks = nrows * (ncols / 128);
        grid = dim3(total_blocks);
        block = dim3(32);
    } else {
        // Simple 32-elem-block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
        const int total_blocks = nrows * (ncols / 32);
        constexpr int THREADS_PER_BLOCK = 256;
        int num_cuda_blocks = (total_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        num_cuda_blocks = min(num_cuda_blocks, 65535);
        grid = dim3(num_cuda_blocks);
        block = dim3(THREADS_PER_BLOCK);
    }

    // Dispatch on the symbolic QType — no more hand-maintained integer literals.
    switch (qtype) {
        case QTYPE_Q4_0:     LAUNCH_REPACK_KERNEL(repack_gemx_q4_0,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q4_1:     LAUNCH_REPACK_KERNEL(repack_gemx_q4_1,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q5_0:     LAUNCH_REPACK_KERNEL(repack_gemx_q5_0,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q5_1:     LAUNCH_REPACK_KERNEL(repack_gemx_q5_1,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q8_0:     LAUNCH_REPACK_KERNEL(repack_gemx_q8_0,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q2_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q2_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q3_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q3_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q4_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q4_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q5_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q5_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q6_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q6_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q8_1:     LAUNCH_REPACK_KERNEL(repack_gemx_q8_1,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_Q8_K:     LAUNCH_REPACK_KERNEL(repack_gemx_q8_K,     grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_QAWQ:     LAUNCH_REPACK_KERNEL(repack_gemx_q_awq,    grid, block, src_data, dst_data, nrows, ncols); break;
        case QTYPE_QAWQ_G64: LAUNCH_REPACK_KERNEL(repack_gemx_q_awq_g64,grid, block, src_data, dst_data, nrows, ncols); break;
        default:
            printf("run_repack_gemx: unknown qtype %d\n", qtype);
            return -1;
    }
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("run_repack_gemx: launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Synchronize to ensure repacking is complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return -1;
    }
    
    return 0;
}

/// Get the size of repacked weights without actually repacking.
///
/// Use this to pre-calculate buffer sizes or verify compatibility.
///
/// Parameters:
/// - nrows: Number of rows in tensor
/// - ncols: Number of columns in tensor
/// - qtype: Quantization type (0-13)
///
/// Returns: Size in bytes of repacked data, or -1 if format not supported
extern "C" int64_t get_repacked_size_bytes(
    int32_t nrows,
    int32_t ncols,
    int32_t qtype
) {
    return get_repacked_size((int)nrows, (int)ncols, (int)qtype);
}

/// Check if a quantization type supports GEMX repacking.
/// (Function name kept for backwards compatibility with FFI)
///
/// Parameters:
/// - qtype: Quantization type (see QType enum)
///
/// Returns: 1 if supported, 0 if not
extern "C" int32_t is_gemx_supported(int32_t qtype) {
    return qtype_output_block_size((int)qtype) > 0 ? 1 : 0;
}
