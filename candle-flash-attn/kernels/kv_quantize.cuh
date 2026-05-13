/*
 * ============================================================================
 * KV CACHE QUANTIZATION KERNELS
 * ============================================================================
 *
 * Quantize FP16/BF16 KV cache to FP8 (E4M3) with dynamic per-head scaling.
 * Designed for SM89+ (Ada Lovelace and later) architectures.
 *
 * Features:
 * - Per-head dynamic scaling for optimal precision
 * - Single kernel quantizes both K and V simultaneously
 * - Scale tracking across tokens for KV cache consistency
 * - Supports HEAD_DIM 64, 128, 256
 *
 * ============================================================================
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// ============================================================================
// QUANTIZATION KERNEL
// ============================================================================

/**
 * Quantize single KV token from FP16/BF16 → FP8 with dynamic scaling.
 *
 * Grid: (batch, n_kv_head)
 * Block: 128 threads
 *
 * @tparam HEAD_DIM  Head dimension (64, 128, 256)
 * @tparam InputT    Input type (half or nv_bfloat16)
 *
 * @param k_in       Input K values [batch, n_kv_head, head_dim]
 * @param v_in       Input V values [batch, n_kv_head, head_dim]
 * @param k_out      Output quantized K [batch, n_kv_head, head_dim]
 * @param v_out      Output quantized V [batch, n_kv_head, head_dim]
 * @param kv_scales  Per-head scales [n_kv_head] - shared K/V scale per head
 * @param batch_size Number of batch elements
 * @param n_kv_head  Number of KV heads
 */
template<int HEAD_DIM, typename InputT = __half>
__global__ void quantize_kv_kernel(
    const InputT* __restrict__ k_in,
    const InputT* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ k_out,
    __nv_fp8_e4m3* __restrict__ v_out,
    float* __restrict__ kv_scales,
    int batch_size,
    int n_kv_head
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= n_kv_head) return;
    
    const int offset = (batch * n_kv_head + head) * HEAD_DIM;
    const InputT* k = k_in + offset;
    const InputT* v = v_in + offset;
    __nv_fp8_e4m3* k_dst = k_out + offset;
    __nv_fp8_e4m3* v_dst = v_out + offset;
    
    // Load to shared memory and find absmax
    __shared__ float k_buf[HEAD_DIM];
    __shared__ float v_buf[HEAD_DIM];
    
    float local_max = 0.f;
    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        float kv, vv;
        if constexpr (std::is_same_v<InputT, __half>) {
            kv = __half2float(k[i]);
            vv = __half2float(v[i]);
        } else {
            kv = __bfloat162float(k[i]);
            vv = __bfloat162float(v[i]);
        }
        k_buf[i] = kv;
        v_buf[i] = vv;
        local_max = fmaxf(local_max, fmaxf(fabsf(kv), fabsf(vv)));
    }
    
    // Warp reduce absmax
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, mask));
    }
    
    // Block reduce via shared memory
    __shared__ float warp_max[4];  // 128 threads = 4 warps
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    
    if (tid < 4) {
        local_max = warp_max[tid];
        for (int mask = 2; mask > 0; mask >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xF, local_max, mask));
        }
        if (tid == 0) warp_max[0] = local_max;
    }
    __syncthreads();
    
    float absmax = warp_max[0];
    // E4M3 max representable value is ~448
    float scale = fmaxf(absmax / 448.f, 1e-12f);
    float inv_scale = 1.f / scale;
    
    // Atomic max update to global scale (track max across all tokens)
    if (tid == 0) {
        // Use atomicMax on the raw bits - not directly available for float,
        // so we use an atomic loop instead
        float old_scale = kv_scales[head];
        while (scale > old_scale) {
            float assumed = old_scale;
            old_scale = atomicCAS(
                reinterpret_cast<unsigned int*>(&kv_scales[head]),
                __float_as_uint(assumed),
                __float_as_uint(scale)
            );
            old_scale = __uint_as_float(old_scale);
            if (old_scale == assumed) break;
        }
    }
    __syncthreads();
    
    // Re-read the (possibly updated) global scale
    scale = kv_scales[head];
    inv_scale = 1.f / scale;
    
    // Quantize using the global scale
    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        float kv = k_buf[i] * inv_scale;
        float vv = v_buf[i] * inv_scale;
        // Clamp to FP8 E4M3 range
        kv = fminf(fmaxf(kv, -448.f), 448.f);
        vv = fminf(fmaxf(vv, -448.f), 448.f);
        __nv_fp8_storage_t k_storage = __nv_cvt_halfraw_to_fp8(__float2half(kv), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t v_storage = __nv_cvt_halfraw_to_fp8(__float2half(vv), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<__nv_fp8_storage_t*>(&k_dst[i]) = k_storage;
        *reinterpret_cast<__nv_fp8_storage_t*>(&v_dst[i]) = v_storage;
    }
#endif
}

// ============================================================================
// DEQUANTIZATION KERNEL  
// ============================================================================

/**
 * Dequantize FP8 KV cache back to FP16.
 *
 * @tparam HEAD_DIM  Head dimension (64, 128, 256)
 */
template<int HEAD_DIM>
__global__ void dequantize_kv_kernel(
    const __nv_fp8_e4m3* __restrict__ k_in,
    const __nv_fp8_e4m3* __restrict__ v_in,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const float* __restrict__ kv_scales,
    int batch_size,
    int n_kv_head
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    const int batch = blockIdx.x;
    const int head = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch >= batch_size || head >= n_kv_head) return;
    
    const float scale = kv_scales[head];
    const int offset = (batch * n_kv_head + head) * HEAD_DIM;
    
    for (int i = tid; i < HEAD_DIM; i += blockDim.x) {
        __nv_fp8_storage_t k_storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&k_in[offset + i]);
        __nv_fp8_storage_t v_storage = *reinterpret_cast<const __nv_fp8_storage_t*>(&v_in[offset + i]);
        
        float kv = __half2float(__nv_cvt_fp8_to_halfraw(k_storage, __NV_E4M3)) * scale;
        float vv = __half2float(__nv_cvt_fp8_to_halfraw(v_storage, __NV_E4M3)) * scale;
        
        k_out[offset + i] = __float2half(kv);
        v_out[offset + i] = __float2half(vv);
    }
#endif
}

// ============================================================================
// LAUNCH WRAPPERS
// ============================================================================

/**
 * Launch KV quantization kernel.
 */
template<typename InputT = __half>
inline void launch_quantize_kv(
    const InputT* k_in, const InputT* v_in,
    __nv_fp8_e4m3* k_out, __nv_fp8_e4m3* v_out,
    float* kv_scales,
    int batch_size, int n_kv_head, int head_dim,
    cudaStream_t stream = 0
) {
    dim3 grid(batch_size, n_kv_head);
    dim3 block(128);
    
    switch (head_dim) {
        case 64:
            quantize_kv_kernel<64, InputT><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        case 128:
            quantize_kv_kernel<128, InputT><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        case 256:
            quantize_kv_kernel<256, InputT><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        default:
            break;
    }
}

/**
 * Launch KV dequantization kernel.
 */
inline void launch_dequantize_kv(
    const __nv_fp8_e4m3* k_in, const __nv_fp8_e4m3* v_in,
    __half* k_out, __half* v_out,
    const float* kv_scales,
    int batch_size, int n_kv_head, int head_dim,
    cudaStream_t stream = 0
) {
    dim3 grid(batch_size, n_kv_head);
    dim3 block(128);
    
    switch (head_dim) {
        case 64:
            dequantize_kv_kernel<64><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        case 128:
            dequantize_kv_kernel<128><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        case 256:
            dequantize_kv_kernel<256><<<grid, block, 0, stream>>>(
                k_in, v_in, k_out, v_out, kv_scales, batch_size, n_kv_head);
            break;
        default:
            break;
    }
}

// ============================================================================
// SCALE INITIALIZATION
// ============================================================================

/**
 * Initialize KV scales to a small value.
 * Call this once per inference session before quantizing any KV tokens.
 */
__global__ void init_kv_scales_kernel(float* kv_scales, int n_kv_head, float init_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_kv_head) {
        kv_scales[idx] = init_val;
    }
}

inline void launch_init_kv_scales(float* kv_scales, int n_kv_head, float init_val = 1e-12f, cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n_kv_head + threads - 1) / threads;
    init_kv_scales_kernel<<<blocks, threads, 0, stream>>>(kv_scales, n_kv_head, init_val);
}
