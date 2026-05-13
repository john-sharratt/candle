// =============================================================================
// FUSED SiLU-MUL KERNEL: out[i] = silu(gate[i]) * up[i]
// =============================================================================
// Fuses the silu activation and element-wise multiply into a single kernel,
// eliminating one kernel launch and one intermediate allocation per call.
// This is the core SwiGLU activation pattern used in modern MoE architectures.
//
// gate = gate_proj(x)   [T, intermediate_dim]
// up   = up_proj(x)     [T, intermediate_dim]
// out  = silu(gate) * up [T, intermediate_dim]   <-- this kernel
//
// Previously this was 2 launches: silu(gate) → tmp, then tmp * up → out.
// Now it's 1 launch with no intermediate allocation.
// =============================================================================

#include "binary_op_macros.cuh"
#include "../fast_exp.cuh"
#include <stdint.h>

// silu_fwd template — same as in unary.cu but needed here since it's not in a header
template<typename T>
__device__ __forceinline__ T fused_silu_fwd(T x) {
    return x / (static_cast<T>(1) + expg(-x));
}

template<>
__device__ __forceinline__ float fused_silu_fwd<float>(float x) {
    return fast_exp::silu<float>(x);
}

template<>
__device__ __forceinline__ double fused_silu_fwd<double>(double x) {
    return x / (1.0 + exp(-x));
}

// =============================================================================
// SCALAR KERNELS (BF16, F16, F32, F8E4M3)
// =============================================================================
// We reuse the BINARY_OP macro: x = gate, y = up, FUNC = silu(x) * y

BINARY_OP(__nv_bfloat16, fused_silu_mul_bf16, __float2bfloat16(fused_silu_fwd(__bfloat162float(x)) * __bfloat162float(y)))
BINARY_OP(__half, fused_silu_mul_f16, __float2half(fused_silu_fwd(__half2float(x)) * __half2float(y)))
BINARY_OP(float, fused_silu_mul_f32, fused_silu_fwd(x) * y)

// FP8: convert to float, compute, convert back
BINARY_OP_NO_LDG(__nv_fp8_e4m3, fused_silu_mul_f8_e4m3,
    __nv_fp8_e4m3(fused_silu_fwd(F8E4M3_TO_FLOAT(x)) * F8E4M3_TO_FLOAT(y)))

// =============================================================================
// VECTORIZED KERNELS
// =============================================================================

// --- BF16 vec2: use bf162 native intrinsics where possible ---
// silu(x) = x * sigmoid(x), but there's no native bf162 silu intrinsic,
// so we compute via float conversion for correctness.
__device__ __forceinline__ __nv_bfloat16 bf_silu_mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    return __float2bfloat16(fused_silu_fwd(fa) * fb);
}

__device__ __forceinline__ __nv_bfloat162 bf2_silu_mul(__nv_bfloat162 a, __nv_bfloat162 b) {
    float fa_lo = __bfloat162float(__low2bfloat16(a));
    float fa_hi = __bfloat162float(__high2bfloat16(a));
    float fb_lo = __bfloat162float(__low2bfloat16(b));
    float fb_hi = __bfloat162float(__high2bfloat16(b));
    return __floats2bfloat162_rn(
        fused_silu_fwd(fa_lo) * fb_lo,
        fused_silu_fwd(fa_hi) * fb_hi
    );
}

BINARY_OP_BF16_VEC2(fused_silu_mul_bf16, bf2_silu_mul, bf_silu_mul)

// --- F16 vec2: use half2 with float conversion for silu ---
__device__ __forceinline__ __half h_silu_mul(__half a, __half b) {
    float fa = __half2float(a);
    float fb = __half2float(b);
    return __float2half(fused_silu_fwd(fa) * fb);
}

__device__ __forceinline__ __half2 h2_silu_mul(__half2 a, __half2 b) {
    float fa_lo = __half2float(__low2half(a));
    float fa_hi = __half2float(__high2half(a));
    float fb_lo = __half2float(__low2half(b));
    float fb_hi = __half2float(__high2half(b));
    return __floats2half2_rn(
        fused_silu_fwd(fa_lo) * fb_lo,
        fused_silu_fwd(fa_hi) * fb_hi
    );
}

BINARY_OP_F16_VEC2(fused_silu_mul_f16, h2_silu_mul, h_silu_mul)

// --- F32 vec4: uses fused_silu_fwd<float> with fast_exp ---
__device__ __forceinline__ float f32_silu_mul(float a, float b) {
    return fused_silu_fwd(a) * b;
}

// Custom vec4 kernel for f32 (can't use BINARY_OP_F32_VEC4 macro which takes
// a simple OP token; we need a function call)
extern "C" __global__ void fused_silu_mul_f32_vec4(
    const unsigned int numel,
    const size_t num_dims,
    const size_t* __restrict__ dims_and_strides,
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out
) {
    // For fused silu_mul, both inputs are always contiguous (fresh GEMM outputs).
    // But we still handle the general case for correctness.
    const size_t *dims = dims_and_strides;
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims;
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims;
    bool lhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, lhs_strides);
    bool rhs_cont = dims_and_strides == nullptr || is_contiguous(num_dims, dims, rhs_strides);

    if (lhs_cont && rhs_cont) {
        bool aligned = ((uintptr_t)lhs % 16 == 0) && ((uintptr_t)rhs % 16 == 0) && ((uintptr_t)out % 16 == 0);

        if (aligned && numel >= 4) {
            const unsigned int vec_numel = numel / 4;
            const unsigned int vec_offset = vec_numel * 4;

            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < vec_numel; i += blockDim.x * gridDim.x) {
                float4 a = ldg_float4(lhs + i * 4);
                float4 b = ldg_float4(rhs + i * 4);
                float4 c;
                c.x = fused_silu_fwd(a.x) * b.x;
                c.y = fused_silu_fwd(a.y) * b.y;
                c.z = fused_silu_fwd(a.z) * b.z;
                c.w = fused_silu_fwd(a.w) * b.w;
                store_float4(out + i * 4, c);
            }

            for (unsigned int i = vec_offset + blockIdx.x * blockDim.x + threadIdx.x;
                 i < numel; i += blockDim.x * gridDim.x) {
                out[i] = fused_silu_fwd(__ldg(lhs + i)) * __ldg(rhs + i);
            }
        } else {
            for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < numel; i += blockDim.x * gridDim.x) {
                out[i] = fused_silu_fwd(__ldg(lhs + i)) * __ldg(rhs + i);
            }
        }
    } else if (lhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i;
            unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) {
                unsigned int dim_val = dims[d];
                unsigned int i_dim = tmp_i % dim_val;
                rhs_i += i_dim * rhs_strides[d];
                tmp_i /= dim_val;
            }
            out[i] = fused_silu_fwd(__ldg(lhs + i)) * __ldg(rhs + rhs_i);
        }
    } else if (rhs_cont) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i;
            unsigned int lhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) {
                unsigned int dim_val = dims[d];
                unsigned int i_dim = tmp_i % dim_val;
                lhs_i += i_dim * lhs_strides[d];
                tmp_i /= dim_val;
            }
            out[i] = fused_silu_fwd(__ldg(lhs + lhs_i)) * __ldg(rhs + i);
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int tmp_i = i;
            unsigned int lhs_i = 0;
            unsigned int rhs_i = 0;
            for (int d = num_dims - 1; d >= 0; d--) {
                unsigned int dim_val = dims[d];
                unsigned int i_dim = tmp_i % dim_val;
                lhs_i += i_dim * lhs_strides[d];
                rhs_i += i_dim * rhs_strides[d];
                tmp_i /= dim_val;
            }
            out[i] = fused_silu_fwd(__ldg(lhs + lhs_i)) * __ldg(rhs + rhs_i);
        }
    }
}

// --- F8E4M3 vec4: 4-byte packed loads ---
__device__ __forceinline__ __nv_fp8_e4m3 f8_silu_mul(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
    return __nv_fp8_e4m3(fused_silu_fwd(F8E4M3_TO_FLOAT(a)) * F8E4M3_TO_FLOAT(b));
}

BINARY_OP_F8E4M3_VEC4(fused_silu_mul_f8_e4m3, f8_silu_mul)
