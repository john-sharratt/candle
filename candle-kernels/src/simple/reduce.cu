#include "cuda_utils.cuh"
#include "warp_reduce.cuh"
#include "vec_utils.cuh"
#include "rope_utils.cuh"
#include "softmax_utils.cuh"
#include "reduce_utils.cuh"
#include "../blocks.cuh"
#include <cmath>
#include <stdint.h>
#include <type_traits>

// IMPORTANT: This must match BLOCK_SIZE in api.cu (256)
const int BLOCK_SIZE = 256;

// =============================================================================
// Type conversion helpers for consistent float conversion across all types
// =============================================================================
template <typename T>
__device__ __forceinline__ float to_float_val(T v) {
    return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T from_float_val(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ float to_float_val(__nv_fp8_e4m3 v) {
    return fp8e4m3_to_float(v);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 from_float_val(float v) {
    return float_to_fp8e4m3(v);
}

// TODO: Maybe add some fast_sum_f16_f32 variant that not only accumulate in f32
// but also expect a f32 output so that this can be used for normalization e.g.
// in softmax.

// Fast reduce sum kernel, this assumes that the dimensions to loop over are at
// the end, each block is responsible for populating one value in the output
// array. There are at most 1024 threads per block.
template <typename T>
__device__ void
fast_sum(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  shr[tid] = 0;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] += src[strided_i];
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] += shr[tid + s];
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

// Template specialization for FP8: accumulates in float32 then converts back
// This avoids the need for FP8 arithmetic operators which don't exist
template <>
__device__ void
fast_sum<__nv_fp8_e4m3>(const size_t src_numel, const size_t el_to_sum_per_block,
                        const size_t num_dims, const size_t *info, 
                        const __nv_fp8_e4m3 *src, __nv_fp8_e4m3 *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  // Accumulate in float32 for precision
  __shared__ float shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  shr[tid] = 0.0f;
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] += fp8e4m3_to_float(src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction in float32
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] += shr[tid + s];
  }

  if (tid == 0)
    dst[dst_id] = float_to_fp8e4m3(shr[0]);
}

// LayerNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L477
template <typename T>
__device__ void layernorm(const T * x, T * dst, const T * alpha, const T * beta, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = to_float_val(x[row*ncols + col]);
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        const int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        // Only warp 0 reduces the per-warp partial sums; other warps supply zero
        float2 warp_val = (warp_id == 0 && lane_id < num_warps)
                          ? s_sum[lane_id]
                          : make_float2(0.0f, 0.0f);
        if (warp_id == 0) {
            mean_var = warp_reduce_sum(warp_val);
        }
        // Broadcast the result from warp 0 thread 0 to all threads
        if (threadIdx.x == 0) {
            s_sum[0] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[0];
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    if (alpha == nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float lhs = (to_float_val(x[row*ncols + col]) - mean) * inv_std; 
          dst[row*ncols + col] = from_float_val<T>(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = to_float_val(beta[col]);
          float lhs = (to_float_val(x[row*ncols + col]) - mean) * inv_std; 
          dst[row*ncols + col] = from_float_val<T>(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = to_float_val(alpha[col]);
          float lhs = (to_float_val(x[row*ncols + col]) - mean) * inv_std; 
          dst[row*ncols + col] = from_float_val<T>(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = to_float_val(alpha[col]);
          float b = to_float_val(beta[col]);
          float lhs = (to_float_val(x[row*ncols + col]) - mean) * inv_std; 
          dst[row*ncols + col] = from_float_val<T>(lhs * a + b);
      }
    }
}

// =============================================================================
// RmsNorm implementation - fully optimized with:
// - Template specialization for compile-time block sizes
// - Shared memory caching (single global read per element) using extern __shared__
// - Vectorized loads/stores (float4 for f32, half2 for f16/bf16)
// - FMA instructions, __ldg for alpha, precomputed 1/ncols
// Adapted from ggml: https://github.com/ggerganov/llama.cpp
// =============================================================================

// NOTE: Warp/block reduction and VecTraits are now in warp_reduce.cuh and vec_utils.cuh

// =============================================================================
// Optimized RmsNorm with shared memory caching and vectorized loads
// Uses extern __shared__ for flexible shared memory allocation at launch time
// =============================================================================

// Fully optimized kernel: shared memory caching + vectorized loads
// This version caches x in shared memory to avoid double reads
// The x_cache pointer is provided from extern __shared__ allocation
template <typename T, int BLOCK_SIZE>
__device__ void rmsnorm_cached(const T * __restrict__ x, T * __restrict__ dst, 
                               const T * __restrict__ alpha, const int ncols, const float eps,
                               float* x_cache) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* x_row = x + row * ncols;
    T* dst_row = dst + row * ncols;
    
    const float inv_ncols = 1.0f / ncols;
    constexpr int VEC_SIZE = VecTraits<T>::VEC_SIZE;
    using VecType = typename VecTraits<T>::VecType;
    
    // === PASS 1: Load, convert to f32, cache, and accumulate sum of squares ===
    float sum_sq = 0.0f;
    
    // Vectorized load path
    const int ncols_vec = (ncols / VEC_SIZE) * VEC_SIZE;
    
    // Process vectorized portion
    #pragma unroll 2
    for (int col = tid * VEC_SIZE; col < ncols_vec; col += BLOCK_SIZE * VEC_SIZE) {
        VecType v = *reinterpret_cast<const VecType*>(&x_row[col]);
        sum_sq += load_and_square_sum(v);
        
        // Cache as f32 - use std::is_same_v for type-safe dispatch
        if constexpr (VEC_SIZE == 4) {
            // float: VEC_SIZE=4
            float4 vf = *reinterpret_cast<const float4*>(&v);
            x_cache[col] = vf.x;
            x_cache[col + 1] = vf.y;
            x_cache[col + 2] = vf.z;
            x_cache[col + 3] = vf.w;
        } else if constexpr (VEC_SIZE == 2) {
            if constexpr (std::is_same_v<T, double>) {
                double2 vd = *reinterpret_cast<const double2*>(&v);
                x_cache[col] = static_cast<float>(vd.x);
                x_cache[col + 1] = static_cast<float>(vd.y);
            } 
            else if constexpr (std::is_same_v<T, __half>) {
                float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&v));
                x_cache[col] = vf.x;
                x_cache[col + 1] = vf.y;
            }
            else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                float2 vf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
                x_cache[col] = vf.x;
                x_cache[col + 1] = vf.y;
            }
        } else {
            x_cache[col] = to_float_val(v);
        }
    }
    
    // Handle remainder (non-vectorized tail)
    for (int col = ncols_vec + tid; col < ncols; col += BLOCK_SIZE) {
        float xi = to_float_val(x_row[col]);
        x_cache[col] = xi;
        sum_sq = __fmaf_rn(xi, xi, sum_sq);
    }
    
    __syncthreads();
    
    // Reduce across block
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    
    // Compute scale factor
    const float scale = rsqrtf(sum_sq * inv_ncols + eps);
    
    // === PASS 2: Write output using cached values (no re-read from global!) ===
    if (alpha == nullptr) {
        // Vectorized store path
        #pragma unroll 2
        for (int col = tid * VEC_SIZE; col < ncols_vec; col += BLOCK_SIZE * VEC_SIZE) {
            if constexpr (VEC_SIZE == 4) {
                store_scaled_vec4(dst_row, x_cache, scale, col);
            } else if constexpr (VEC_SIZE == 2) {
                if constexpr (std::is_same_v<T, double>) {
                    store_scaled_vec2(dst_row, x_cache, scale, col);
                }
                else if constexpr (std::is_same_v<T, __half>) {
                    store_scaled_vec2(dst_row, x_cache, scale, col);
                }
                else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    store_scaled_vec2(dst_row, x_cache, scale, col);
                }
            } else {
                store_scaled(dst_row, x_cache, scale, col);
            }
        }
        
        // Handle remainder
        for (int col = ncols_vec + tid; col < ncols; col += BLOCK_SIZE) {
            store_scaled(dst_row, x_cache, scale, col);
        }
    } else {
        // With alpha: need to load alpha values
        // Use __ldg for alpha (it's read-only and benefits from texture cache)
        #pragma unroll 2
        for (int col = tid * VEC_SIZE; col < ncols_vec; col += BLOCK_SIZE * VEC_SIZE) {
            // Load alpha and apply
            if constexpr (VEC_SIZE == 4) {
                float4 a4;
                a4.x = static_cast<float>(__ldg(&alpha[col]));
                a4.y = static_cast<float>(__ldg(&alpha[col + 1]));
                a4.z = static_cast<float>(__ldg(&alpha[col + 2]));
                a4.w = static_cast<float>(__ldg(&alpha[col + 3]));
                float4 out;
                out.x = scale * x_cache[col] * a4.x;
                out.y = scale * x_cache[col + 1] * a4.y;
                out.z = scale * x_cache[col + 2] * a4.z;
                out.w = scale * x_cache[col + 3] * a4.w;
                *reinterpret_cast<float4*>(&dst_row[col]) = out;
            } else if constexpr (VEC_SIZE == 2) {
                float a0 = static_cast<float>(__ldg(&alpha[col]));
                float a1 = static_cast<float>(__ldg(&alpha[col + 1]));
                float v0 = scale * x_cache[col] * a0;
                float v1 = scale * x_cache[col + 1] * a1;
                
                if constexpr (std::is_same_v<T, double>) {
                    double2 out;
                    out.x = static_cast<double>(v0);
                    out.y = static_cast<double>(v1);
                    *reinterpret_cast<double2*>(&dst_row[col]) = out;
                }
                else if constexpr (std::is_same_v<T, __half>) {
                    __half2 out = __floats2half2_rn(v0, v1);
                    *reinterpret_cast<__half2*>(&dst_row[col]) = out;
                }
                else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    __nv_bfloat162 out = __floats2bfloat162_rn(v0, v1);
                    *reinterpret_cast<__nv_bfloat162*>(&dst_row[col]) = out;
                }
            } else {
                float a = to_float_val(__ldg(&alpha[col]));
                dst_row[col] = from_float_val<T>(scale * x_cache[col] * a);
            }
        }
        
        // Handle remainder
        for (int col = ncols_vec + tid; col < ncols; col += BLOCK_SIZE) {
            float a = to_float_val(__ldg(&alpha[col]));
            dst_row[col] = from_float_val<T>(scale * x_cache[col] * a);
        }
    }
}

// Fallback non-cached version for large ncols (>8192) or dynamic block sizes
template <typename T, int BLOCK_SIZE>
__device__ void rmsnorm_uncached(const T * __restrict__ x, T * __restrict__ dst, 
                                  const T * __restrict__ alpha, const int ncols, const float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* x_row = x + row * ncols;
    T* dst_row = dst + row * ncols;
    
    const float inv_ncols = 1.0f / ncols;
    
    // === PASS 1: Compute sum of squares ===
    float sum_sq = 0.0f;
    
    #pragma unroll 4
    for (int col = tid; col < ncols; col += BLOCK_SIZE) {
        float xi = to_float_val(x_row[col]);
        sum_sq = __fmaf_rn(xi, xi, sum_sq);
    }
    
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    const float scale = rsqrtf(sum_sq * inv_ncols + eps);
    
    // === PASS 2: Normalize and write ===
    if (alpha == nullptr) {
        #pragma unroll 4
        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            float xi = to_float_val(x_row[col]);
            dst_row[col] = from_float_val<T>(scale * xi);
        }
    } else {
        #pragma unroll 4
        for (int col = tid; col < ncols; col += BLOCK_SIZE) {
            float xi = to_float_val(x_row[col]);
            float a = to_float_val(__ldg(&alpha[col]));
            dst_row[col] = from_float_val<T>(scale * xi * a);
        }
    }
}

// Dynamic block size version (for non-standard sizes)
// Uses extern __shared__ for flexible allocation
template <typename T>
__device__ void rmsnorm_dynamic(const T * __restrict__ x, T * __restrict__ dst,
                                const T * __restrict__ alpha, const int ncols, 
                                const int block_size, const float eps,
                                float* x_cache, bool use_cache) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* x_row = x + row * ncols;
    T* dst_row = dst + row * ncols;
    
    const float inv_ncols = 1.0f / ncols;
    
    float sum_sq = 0.0f;
    
    if (use_cache && x_cache != nullptr) {
        // Cached path: load data once and store in shared memory
        for (int col = tid; col < ncols; col += block_size) {
            float xi = to_float_val(x_row[col]);
            x_cache[col] = xi;
            sum_sq = __fmaf_rn(xi, xi, sum_sq);
        }
        __syncthreads();
        
        sum_sq = block_reduce_sum_dynamic(sum_sq, block_size);
        const float scale = rsqrtf(sum_sq * inv_ncols + eps);
        
        if (alpha == nullptr) {
            for (int col = tid; col < ncols; col += block_size) {
                dst_row[col] = from_float_val<T>(scale * x_cache[col]);
            }
        } else {
            for (int col = tid; col < ncols; col += block_size) {
                float a = to_float_val(__ldg(&alpha[col]));
                dst_row[col] = from_float_val<T>(scale * x_cache[col] * a);
            }
        }
    } else {
        // Uncached path: read data twice from global memory
        for (int col = tid; col < ncols; col += block_size) {
            float xi = to_float_val(x_row[col]);
            sum_sq = __fmaf_rn(xi, xi, sum_sq);
        }
        
        sum_sq = block_reduce_sum_dynamic(sum_sq, block_size);
        const float scale = rsqrtf(sum_sq * inv_ncols + eps);
        
        if (alpha == nullptr) {
            for (int col = tid; col < ncols; col += block_size) {
                float xi = to_float_val(x_row[col]);
                dst_row[col] = from_float_val<T>(scale * xi);
            }
        } else {
            for (int col = tid; col < ncols; col += block_size) {
                float xi = to_float_val(x_row[col]);
                float a = to_float_val(__ldg(&alpha[col]));
                dst_row[col] = from_float_val<T>(scale * xi * a);
            }
        }
    }
}

// Maximum columns we can cache in shared memory (48KB / 4 bytes = 12288, but use 8192 for safety)
#define MAX_CACHED_COLS 8192

// Optimized entry point with cache selection
// x_cache is provided via extern __shared__ from the kernel launch
template <typename T, int BLOCK_SIZE>
__device__ void rmsnorm_opt(const T * __restrict__ x, T * __restrict__ dst, 
                            const T * __restrict__ alpha, const int ncols, const float eps,
                            float* x_cache) {
    // Use shared memory cache for sizes that fit
    if (ncols <= MAX_CACHED_COLS && x_cache != nullptr) {
        rmsnorm_cached<T, BLOCK_SIZE>(x, dst, alpha, ncols, eps, x_cache);
    } else {
        // Fall back to uncached version for very large ncols
        rmsnorm_uncached<T, BLOCK_SIZE>(x, dst, alpha, ncols, eps);
    }
}

// Main entry point: dispatches to specialized versions
// Uses extern __shared__ for dynamic shared memory allocation
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
    // Use extern __shared__ for flexible dynamic allocation
    // The dispatcher must allocate ncols * sizeof(float) bytes when ncols <= MAX_CACHED_COLS
    extern __shared__ float shared_cache[];
    
    // Determine if we should use cache (only if shared memory was allocated)
    const bool use_cache = (ncols <= MAX_CACHED_COLS);
    float* x_cache = use_cache ? shared_cache : nullptr;
    
    // Dispatch to compile-time specialized versions for common sizes
    // This eliminates runtime conditionals in the hot path
    switch (block_size) {
        case 32:   rmsnorm_opt<T, 32>(x, dst, alpha, ncols, eps, x_cache); break;
        case 64:   rmsnorm_opt<T, 64>(x, dst, alpha, ncols, eps, x_cache); break;
        case 128:  rmsnorm_opt<T, 128>(x, dst, alpha, ncols, eps, x_cache); break;
        case 256:  rmsnorm_opt<T, 256>(x, dst, alpha, ncols, eps, x_cache); break;
        case 512:  rmsnorm_opt<T, 512>(x, dst, alpha, ncols, eps, x_cache); break;
        case 1024: rmsnorm_opt<T, 1024>(x, dst, alpha, ncols, eps, x_cache); break;
        default:   rmsnorm_dynamic<T>(x, dst, alpha, ncols, block_size, eps, x_cache, use_cache); break;
    }
}

// =============================================================================
// FUSED RMSNORM → q8a128 (producer epilogue B1/B3/B5)
// =============================================================================
// One block per row. PASS 1 caches the row in shared memory and block-reduces
// Σx² → scale = rsqrt(Σx²/ncols + eps) (same formula as `rmsnorm`). PASS 2 lays
// the block out as one warp per 128-tile (lane owns 4 contiguous elements) and
// writes the q8a128 activation block directly — the per-128 amax/Σx butterfly +
// char4 store + lane-0 ds, identical to `quantize_q8a128_kernel`, but on the
// normalized values. Each normalized value is round-tripped through the store
// dtype T so the result tracks `rmsnorm`→`quantize_q8a128` within float margin.
// Output is the flat-grouped q8a1024 buffer (see blocks.cuh). Requires
// ncols % 128 == 0 and alpha != nullptr (the RMSNorm weight).

// Vectorized 4-wide load (mirrors quantize_q8a128's q8a128_load4): one 16/8-byte vector load → 4
// floats, for the PASS-2 alpha read. PASS-1 uses the VecTraits path (load_and_square_sum).
template <typename T>
__device__ __forceinline__ void rms_load4(const T* p, float& a, float& b, float& c, float& d);
template <>
__device__ __forceinline__ void rms_load4<float>(const float* p, float& a, float& b, float& c, float& d) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    a = v.x; b = v.y; c = v.z; d = v.w;
}
template <>
__device__ __forceinline__ void rms_load4<__half>(const __half* p, float& a, float& b, float& c, float& d) {
    const __half2* h = reinterpret_cast<const __half2*>(p);
    const float2 lo = __half22float2(h[0]);
    const float2 hi = __half22float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}
template <>
__device__ __forceinline__ void rms_load4<__nv_bfloat16>(const __nv_bfloat16* p, float& a, float& b, float& c, float& d) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(p);
    const float2 lo = __bfloat1622float2(h[0]);
    const float2 hi = __bfloat1622float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}

template <typename T, int BLOCK_SIZE>
__device__ void rmsnorm_q8a128_impl(
    const T* __restrict__ x, block_q8a128* __restrict__ out,
    const T* __restrict__ alpha, const int ncols, const float eps, float* x_cache)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* x_row = x + (int64_t)row * ncols;
    const float inv_ncols = 1.0f / ncols;

    // PASS 1: cache the row as f32 and accumulate Σx² (vectorized 16-byte loads, same path as the
    // FP `rmsnorm_cached`). ncols % 128 == 0 so the vector loop covers the whole row.
    constexpr int VEC_SIZE = VecTraits<T>::VEC_SIZE;
    using VecType = typename VecTraits<T>::VecType;
    const int ncols_vec = (ncols / VEC_SIZE) * VEC_SIZE;
    float sum_sq = 0.0f;
    #pragma unroll 2
    for (int col = tid * VEC_SIZE; col < ncols_vec; col += BLOCK_SIZE * VEC_SIZE) {
        VecType v = *reinterpret_cast<const VecType*>(&x_row[col]);
        sum_sq += load_and_square_sum(v);
        if constexpr (VEC_SIZE == 4) {
            const float4 vf = *reinterpret_cast<const float4*>(&v);
            x_cache[col] = vf.x;
            x_cache[col + 1] = vf.y;
            x_cache[col + 2] = vf.z;
            x_cache[col + 3] = vf.w;
        } else if constexpr (VEC_SIZE == 2 && std::is_same_v<T, __half>) {
            const float2 vf = __half22float2(*reinterpret_cast<const __half2*>(&v));
            x_cache[col] = vf.x;
            x_cache[col + 1] = vf.y;
        } else if constexpr (VEC_SIZE == 2 && std::is_same_v<T, __nv_bfloat16>) {
            const float2 vf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
            x_cache[col] = vf.x;
            x_cache[col + 1] = vf.y;
        } else {
            x_cache[col] = to_float_val(v);
        }
    }
    // Tail (only if ncols is not a VEC_SIZE multiple — never for the q8a128 contract, kept safe).
    for (int col = ncols_vec + tid; col < ncols; col += BLOCK_SIZE) {
        float xi = to_float_val(x_row[col]);
        x_cache[col] = xi;
        sum_sq = __fmaf_rn(xi, xi, sum_sq);
    }
    __syncthreads();
    sum_sq = block_reduce_sum<BLOCK_SIZE>(sum_sq);
    const float scale = rsqrtf(sum_sq * inv_ncols + eps);
    __syncthreads(); // x_cache must be fully visible to every warp before PASS 2

    // PASS 2: one warp per 128-tile; lane owns 4 contiguous normalized elements.
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = BLOCK_SIZE >> 5;
    const int tiles_per_row = ncols >> 7; // ncols / 128
    uint8_t* obytes = reinterpret_cast<uint8_t*>(out);
    for (int t = warp; t < tiles_per_row; t += n_warps) {
        const int col0 = t * 128 + lane * 4;
        // normalize, fold alpha (vector-loaded), round through T (mirrors the FP store rmsnorm does).
        float a0, a1, a2, a3;
        rms_load4<T>(alpha + col0, a0, a1, a2, a3);
        const float n0 = to_float_val(from_float_val<T>(scale * x_cache[col0 + 0] * a0));
        const float n1 = to_float_val(from_float_val<T>(scale * x_cache[col0 + 1] * a1));
        const float n2 = to_float_val(from_float_val<T>(scale * x_cache[col0 + 2] * a2));
        const float n3 = to_float_val(from_float_val<T>(scale * x_cache[col0 + 3] * a3));

        float amax = fmaxf(fmaxf(fabsf(n0), fabsf(n1)), fmaxf(fabsf(n2), fabsf(n3)));
        float s = n0 + n1 + n2 + n3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
            s += __shfl_xor_sync(0xffffffff, s, off, 32);
        }
        const float id = (amax != 0.f) ? 127.f / amax : 0.f;
        const int64_t flat = (int64_t)row * tiles_per_row + t;
        *reinterpret_cast<char4*>(obytes + q8a1024_qs_off(flat) + lane * 4) = make_char4(
            (int8_t)__float2int_rn(n0 * id),
            (int8_t)__float2int_rn(n1 * id),
            (int8_t)__float2int_rn(n2 * id),
            (int8_t)__float2int_rn(n3 * id));
        if (lane == 0) {
            half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(flat));
            ds[0] = make_half2(__float2half_rn(amax / 127.f), __float2half_rn(s));
        }
    }
}

template <typename T>
__device__ void rmsnorm_q8a128(
    const T* x, block_q8a128* out, const T* alpha,
    const int ncols, const int block_size, const float eps)
{
    extern __shared__ float shared_cache[];
    switch (block_size) {
        case 32:   rmsnorm_q8a128_impl<T, 32>(x, out, alpha, ncols, eps, shared_cache); break;
        case 64:   rmsnorm_q8a128_impl<T, 64>(x, out, alpha, ncols, eps, shared_cache); break;
        case 128:  rmsnorm_q8a128_impl<T, 128>(x, out, alpha, ncols, eps, shared_cache); break;
        case 256:  rmsnorm_q8a128_impl<T, 256>(x, out, alpha, ncols, eps, shared_cache); break;
        case 512:  rmsnorm_q8a128_impl<T, 512>(x, out, alpha, ncols, eps, shared_cache); break;
        default:   rmsnorm_q8a128_impl<T, 1024>(x, out, alpha, ncols, eps, shared_cache); break;
    }
}

// =============================================================================
// SOFTMAX - OPTIMIZED VERSION
// =============================================================================
// Key optimizations over ggml baseline:
//   1. CRITICAL BUG FIX: Proper inter-warp reduction (old code only worked for 32 threads)
//   2. Online algorithm: single-pass max + sum with correction factor
//   3. Vectorized memory access (float4, half2, bfloat162)
//   4. __ldg() for read-only input (texture cache)
//   5. __restrict__ qualifiers for pointer aliasing hints
//   6. Fast math intrinsics (__expf, __frcp_rn instead of division)
//   7. #pragma unroll for critical loops
//   8. Type specializations for float, double, half, bfloat16
// See softmax_utils.cuh for implementation details.
// =============================================================================

template <typename T, typename ACC>
__device__ void softmax(const T* __restrict__ x, T* __restrict__ dst, const int ncols) {
    softmax_register_based<T, ACC>(x, dst, ncols);
}

// =============================================================================
// ROPE (ROTARY POSITION EMBEDDING) KERNELS - OPTIMIZED
// =============================================================================
// Key optimizations:
//   1. __restrict__ qualifiers for pointer aliasing hints
//   2. __ldg() for read-only cos/sin arrays (texture cache)
//   3. FMA instructions for rotation (fused multiply-add)
//   4. Precomputed half dimensions to eliminate division
//   5. Vectorized float2 loads for interleaved layout
//   6. Boundary check at start with early return
// =============================================================================

// ropei: Interleaved layout [..., (x, x'), (y, y'), ...]
// Pairs are adjacent, allowing vectorized loads
template <typename T>
__device__ void ropei(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t stride_b
) {
    // Precompute to avoid division in hot path
    const uint32_t half_td = td >> 1;  // td / 2
    const uint32_t total_pairs = bh * half_td;
    const uint32_t stride_b_pairs = (stride_b > 0) ? (stride_b >> 1) : 0;
    
    rope_interleaved_optimized(src, cos, sin, dst, total_pairs, half_td, stride_b_pairs);
}

// rope: Rotary layout [..., x0..x_{d/2-1}, x'0..x'_{d/2-1}, ...]
// Pairs separated by d/2 elements
template <typename T>
__device__ void rope(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t bh,
    const uint32_t td,
    const uint32_t d,
    const uint32_t stride_b
) {
    // Precompute divisions
    const uint32_t half_td = td >> 1;
    const uint32_t half_d = d >> 1;
    const uint32_t total_pairs = bh * half_td;
    const uint32_t stride_b_pairs = (stride_b > 0) ? (stride_b >> 1) : 0;
    
    rope_rotary_optimized(src, cos, sin, dst, total_pairs, half_td, half_d, d, stride_b_pairs);
}

// rope_thd: THD layout [batch, time, head, dim]
template <typename T>
__device__ void rope_thd(
    const T* __restrict__ src,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ dst,
    const uint32_t b,
    const uint32_t t,
    const uint32_t h,
    const uint32_t d,
    const uint32_t stride_b
) {
    // Precompute divisions
    const uint32_t half_d = d >> 1;
    const uint32_t half_td = (t * d) >> 1;
    const uint32_t total_pairs = b * t * h * half_d;
    const uint32_t stride_b_pairs = (stride_b > 0) ? (stride_b >> 1) : 0;
    
    rope_thd_optimized(src, cos, sin, dst, total_pairs, half_d, d, h, t, half_td, stride_b_pairs);
}

// =============================================================================
// FAST_MAX - Optimized with warp shuffle reduction and vectorized loads
// =============================================================================
// Key optimizations:
//   1. Warp shuffle for intra-warp reduction (3-5× faster than shared memory)
//   2. Single __syncthreads() for entire block (vs 8+ in original)
//   3. Vectorized float4 loads for contiguous tensors
//   4. Multiple accumulators for instruction-level parallelism
//   5. __ldg() for read-only source array
//   6. __restrict__ pointers for aliasing hints
// =============================================================================

template <typename T>
__device__ void
fast_max(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t* __restrict__ info, 
         const T* __restrict__ src, T* __restrict__ dst) {
    const size_t* dims = info;
    const size_t* strides = info + num_dims;

    const unsigned int tid = threadIdx.x;
    const unsigned int dst_id = blockIdx.x;

    const unsigned int start_idx = dst_id * el_to_sum_per_block;
    const unsigned int stop_idx = min(start_idx + (unsigned int)el_to_sum_per_block, (unsigned int)src_numel);

    T max_val;

    if constexpr (std::is_same<T, float>::value) {
        // Check for contiguous fast path (last dimension has stride 1)
        const bool is_contiguous = (num_dims > 0 && strides[num_dims - 1] == 1);
        if (is_contiguous) {
            // Contiguous float path with vectorized loads
            max_val = fast_max_contiguous_f32<BLOCK_SIZE>(
                reinterpret_cast<const float*>(src), start_idx, stop_idx);
        } else {
            max_val = fast_max_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides, start_idx, stop_idx);
        }
    } else {
        // Strided path for non-contiguous or non-float types
        max_val = fast_max_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides, start_idx, stop_idx);
    }

    if (tid == 0) {
        dst[dst_id] = max_val;
    }
}

// =============================================================================
// FAST_MIN - Optimized with warp shuffle reduction and vectorized loads
// =============================================================================

template <typename T>
__device__ void
fast_min(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t* __restrict__ info,
         const T* __restrict__ src, T* __restrict__ dst) {
    const size_t* dims = info;
    const size_t* strides = info + num_dims;

    const unsigned int tid = threadIdx.x;
    const unsigned int dst_id = blockIdx.x;

    const unsigned int start_idx = dst_id * el_to_sum_per_block;
    const unsigned int stop_idx = min(start_idx + (unsigned int)el_to_sum_per_block, (unsigned int)src_numel);

    T min_val;

    if constexpr (std::is_same<T, float>::value) {
        // Check for contiguous fast path (last dimension has stride 1)
        const bool is_contiguous = (num_dims > 0 && strides[num_dims - 1] == 1);
        if (is_contiguous) {
            min_val = fast_min_contiguous_f32<BLOCK_SIZE>(
                reinterpret_cast<const float*>(src), start_idx, stop_idx);
        } else {
            min_val = fast_min_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides, start_idx, stop_idx);
        }
    } else {
        min_val = fast_min_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides, start_idx, stop_idx);
    }

    if (tid == 0) {
        dst[dst_id] = min_val;
    }
}

// =============================================================================
// FAST_ARGMIN - Optimized with warp shuffle, precomputed last_dim
// =============================================================================
// Additional optimizations:
//   7. Precomputed last_dim to avoid expensive modulo operation
//   8. Combined value+index reduction in single warp shuffle pass
// =============================================================================

template <typename T>
__device__ void
fast_argmin(const size_t src_numel, const size_t el_to_sum_per_block,
            const size_t num_dims, const size_t* __restrict__ info,
            const T* __restrict__ src, uint32_t* __restrict__ dst) {
    const size_t* dims = info;
    const size_t* strides = info + num_dims;

    const unsigned int tid = threadIdx.x;
    const unsigned int dst_id = blockIdx.x;

    const unsigned int start_idx = dst_id * el_to_sum_per_block;
    const unsigned int stop_idx = min(start_idx + (unsigned int)el_to_sum_per_block, (unsigned int)src_numel);

    // Precompute last_dim to avoid expensive modulo in hot loop
    const unsigned int last_dim = (num_dims > 0) ? dims[num_dims - 1] : 1;
    const bool is_contiguous = (num_dims > 0 && strides[num_dims - 1] == 1);

    T min_val;
    uint32_t min_idx;

    if (is_contiguous && std::is_same<T, float>::value) {
        fast_argmin_contiguous_f32<BLOCK_SIZE>(
            reinterpret_cast<const float*>(src), start_idx, stop_idx, last_dim,
            *reinterpret_cast<float*>(&min_val), min_idx);
    } else {
        fast_argmin_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides, 
                                            start_idx, stop_idx, last_dim,
                                            min_val, min_idx);
    }

    if (tid == 0) {
        dst[dst_id] = min_idx;
    }
}

// =============================================================================
// FAST_ARGMAX - Optimized with warp shuffle, precomputed last_dim
// =============================================================================

template <typename T>
__device__ void
fast_argmax(const size_t src_numel, const size_t el_to_sum_per_block,
            const size_t num_dims, const size_t* __restrict__ info,
            const T* __restrict__ src, uint32_t* __restrict__ dst) {
    const size_t* dims = info;
    const size_t* strides = info + num_dims;

    const unsigned int tid = threadIdx.x;
    const unsigned int dst_id = blockIdx.x;

    const unsigned int start_idx = dst_id * el_to_sum_per_block;
    const unsigned int stop_idx = min(start_idx + (unsigned int)el_to_sum_per_block, (unsigned int)src_numel);

    const unsigned int last_dim = (num_dims > 0) ? dims[num_dims - 1] : 1;
    const bool is_contiguous = (num_dims > 0 && strides[num_dims - 1] == 1);

    T max_val;
    uint32_t max_idx;

    if (is_contiguous && std::is_same<T, float>::value) {
        fast_argmax_contiguous_f32<BLOCK_SIZE>(
            reinterpret_cast<const float*>(src), start_idx, stop_idx, last_dim,
            *reinterpret_cast<float*>(&max_val), max_idx);
    } else {
        fast_argmax_strided<T, BLOCK_SIZE>(src, num_dims, dims, strides,
                                            start_idx, stop_idx, last_dim,
                                            max_val, max_idx);
    }

    if (tid == 0) {
        dst[dst_id] = max_idx;
    }
}

#define FAST_OP(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME, SUM_NAME) \
  extern "C" __global__ void ARGMIN_NAME(                                      \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void ARGMAX_NAME(                                     \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void MIN_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void MAX_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void SUM_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }

// FAST_OP without fast_sum - for types without atomicAdd (e.g. FP8)
#define FAST_OP_NO_SUM(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME) \
  extern "C" __global__ void ARGMIN_NAME(                                      \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void ARGMAX_NAME(                                     \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void MIN_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void MAX_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }

#define SUM_OP(TYPENAME, FN_NAME)                                              \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t num_sum_dims,    \
      const size_t *info, const TYPENAME *inp, TYPENAME *out) {                \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    const size_t *sum_dims_l = info + 2 * num_dims;                            \
    const size_t *sum_dims_s = info + 2 * num_dims + num_sum_dims;             \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[i]);                                    \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i]);                            \
      }                                                                        \
    }                                                                          \
  }

#define SOFTMAX_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst,                                      \
      const int n_cols) {                                                      \
    softmax<TYPENAME, ACC_TYPENAME>(src, dst, n_cols);                         \
  }                                                                            \

#define RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const int n_cols, const int block_size, const float eps) {               \
    rmsnorm<TYPENAME>(src, dst, alpha, n_cols, block_size, eps);               \
  }                                                                            \

#define RMSNORM_Q8A128_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, void *out, const TYPENAME *alpha,                   \
      const int n_cols, const int block_size, const float eps) {              \
    rmsnorm_q8a128<TYPENAME>(                                                  \
        src, reinterpret_cast<block_q8a128*>(out), alpha, n_cols, block_size, eps); \
  }                                                                            \

#define LAYERNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const TYPENAME *beta, const int n_cols, const int block_size, const float eps) { \
    layernorm<TYPENAME>(src, dst, alpha, beta, n_cols, block_size, eps);       \
  }                                                                            \

#define ROPE_OP(TYPENAME, FN_NAME, FN_NAME_I, FN_NAME_THD) \
  extern "C" __global__ void FN_NAME_I( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t stride_b) { \
    ropei<TYPENAME>(src, cos, sin, dst, bh, td, stride_b); \
  } \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope<TYPENAME>(src, cos, sin, dst, bh, td, d, stride_b); \
  } \
  extern "C" __global__ void FN_NAME_THD( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t b, \
      const uint32_t t, \
      const uint32_t h, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope_thd<TYPENAME>(src, cos, sin, dst, b, t, h, d, stride_b); \
  } \

SOFTMAX_OP(__nv_bfloat16, float, softmax_bf16)
RMSNORM_OP(__nv_bfloat16, rmsnorm_bf16)
RMSNORM_Q8A128_OP(__nv_bfloat16, rmsnorm_q8a128_bf16)
LAYERNORM_OP(__nv_bfloat16, layernorm_bf16)
ROPE_OP(__nv_bfloat16, rope_bf16, rope_i_bf16, rope_thd_bf16)
SUM_OP(__nv_bfloat16, sum_bf16)
FAST_OP(__nv_bfloat16, fast_min_bf16, fast_max_bf16, fast_argmin_bf16, fast_argmax_bf16, fast_sum_bf16)

// FP8E4M3 ops - NOTE: SUM_OP not supported (no atomicAdd for FP8)
SOFTMAX_OP(__nv_fp8_e4m3, float, softmax_f8_e4m3)
RMSNORM_OP(__nv_fp8_e4m3, rmsnorm_f8_e4m3)
LAYERNORM_OP(__nv_fp8_e4m3, layernorm_f8_e4m3)
ROPE_OP(__nv_fp8_e4m3, rope_f8_e4m3, rope_i_f8_e4m3, rope_thd_f8_e4m3)
FAST_OP_NO_SUM(__nv_fp8_e4m3, fast_min_f8_e4m3, fast_max_f8_e4m3, fast_argmin_f8_e4m3, fast_argmax_f8_e4m3)
// FP8 fast_sum uses the template specialization with float32 accumulation
extern "C" __global__ void fast_sum_f8_e4m3(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_fp8_e4m3 *src,
    __nv_fp8_e4m3 *dst) {
  fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}

SOFTMAX_OP(__half, float, softmax_f16)
RMSNORM_OP(__half, rmsnorm_f16)
RMSNORM_Q8A128_OP(__half, rmsnorm_q8a128_f16)
LAYERNORM_OP(__half, layernorm_f16)
ROPE_OP(__half, rope_f16, rope_i_f16, rope_thd_f16)
SUM_OP(__half, sum_f16)
FAST_OP(__half, fast_min_f16, fast_max_f16, fast_argmin_f16, fast_argmax_f16, fast_sum_f16)

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)
SOFTMAX_OP(float, float, softmax_f32)
SOFTMAX_OP(double, double, softmax_f64)
RMSNORM_OP(float, rmsnorm_f32)
RMSNORM_Q8A128_OP(float, rmsnorm_q8a128_f32)
RMSNORM_OP(double, rmsnorm_f64)
LAYERNORM_OP(float, layernorm_f32)
LAYERNORM_OP(double, layernorm_f64)
ROPE_OP(float, rope_f32, rope_i_f32, rope_thd_f32)
ROPE_OP(double, rope_f64, rope_i_f64, rope_thd_f64)

FAST_OP(float, fast_min_f32, fast_max_f32, fast_argmin_f32, fast_argmax_f32, fast_sum_f32)
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)
