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
#include "../blocks.cuh"
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

// =============================================================================
// FUSED SwiGLU → q8a128 (producer epilogue B4)
// =============================================================================
// out[i] = silu(gate[i]) * up[i], quantized directly to q8a128 — one kernel that
// replaces silu_mul (FP store) + quantize_acts_q8a128 (re-read). One warp per
// 128-tile: lane owns 4 contiguous elements; the SwiGLU result is rounded through
// the store dtype T (mirrors the FP store), then the per-128 amax/Σx butterfly +
// char4 store + lane-0 ds write the q8a1024 flat-grouped block (see blocks.cuh),
// identical to quantize_q8a128_kernel on the SwiGLU output. silu uses the same
// fused_silu_fwd<float> (fast_exp) as the unfused kernel, so the result tracks the
// two-call path within float margin. Requires (rows*cols) % 128 == 0.

template <typename T>
__device__ __forceinline__ void smq8_load4(const T* p, float& a, float& b, float& c, float& d);
template <>
__device__ __forceinline__ void smq8_load4<float>(const float* p, float& a, float& b, float& c, float& d) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    a = v.x; b = v.y; c = v.z; d = v.w;
}
template <>
__device__ __forceinline__ void smq8_load4<__half>(const __half* p, float& a, float& b, float& c, float& d) {
    const __half2* h = reinterpret_cast<const __half2*>(p);
    const float2 lo = __half22float2(h[0]);
    const float2 hi = __half22float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}
template <>
__device__ __forceinline__ void smq8_load4<__nv_bfloat16>(const __nv_bfloat16* p, float& a, float& b, float& c, float& d) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(p);
    const float2 lo = __bfloat1622float2(h[0]);
    const float2 hi = __bfloat1622float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}

// Round a float through the store dtype T and back (mirrors silu_mul's FP store).
template <typename T> __device__ __forceinline__ float smq8_round(float v);
template <> __device__ __forceinline__ float smq8_round<float>(float v) { return v; }
template <> __device__ __forceinline__ float smq8_round<__half>(float v) { return __half2float(__float2half_rn(v)); }
template <> __device__ __forceinline__ float smq8_round<__nv_bfloat16>(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

template <typename T>
__device__ void silu_mul_q8a128_impl(
    const T* __restrict__ gate, const T* __restrict__ up,
    block_q8a128* __restrict__ out, int rows, int cols)
{
    const int total_tiles = (int)(((int64_t)rows * cols) / 128);
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    uint8_t* obytes = reinterpret_cast<uint8_t*>(out);
    for (int tile = warp; tile < total_tiles; tile += total_warps) {
        const int64_t base = (int64_t)tile * 128 + (int64_t)lane * 4;
        float g0, g1, g2, g3; smq8_load4<T>(gate + base, g0, g1, g2, g3);
        float u0, u1, u2, u3; smq8_load4<T>(up + base, u0, u1, u2, u3);
        const float n0 = smq8_round<T>(fused_silu_fwd<float>(g0) * u0);
        const float n1 = smq8_round<T>(fused_silu_fwd<float>(g1) * u1);
        const float n2 = smq8_round<T>(fused_silu_fwd<float>(g2) * u2);
        const float n3 = smq8_round<T>(fused_silu_fwd<float>(g3) * u3);

        float amax = fmaxf(fmaxf(fabsf(n0), fabsf(n1)), fmaxf(fabsf(n2), fabsf(n3)));
        float s = n0 + n1 + n2 + n3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
            s += __shfl_xor_sync(0xffffffff, s, off, 32);
        }
        const float id = (amax != 0.f) ? 127.f / amax : 0.f;
        *reinterpret_cast<char4*>(obytes + q8a1024_qs_off(tile) + lane * 4) = make_char4(
            (int8_t)__float2int_rn(n0 * id),
            (int8_t)__float2int_rn(n1 * id),
            (int8_t)__float2int_rn(n2 * id),
            (int8_t)__float2int_rn(n3 * id));
        if (lane == 0) {
            half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(tile));
            ds[0] = make_half2(__float2half_rn(amax / 127.f), __float2half_rn(s));
        }
    }
}

#define SILU_MUL_Q8A128_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME* gate, const TYPENAME* up, void* out, int rows, int cols) { \
    silu_mul_q8a128_impl<TYPENAME>(gate, up, reinterpret_cast<block_q8a128*>(out), rows, cols); \
  }

SILU_MUL_Q8A128_OP(float, silu_mul_q8a128_f32)
SILU_MUL_Q8A128_OP(__half, silu_mul_q8a128_f16)
SILU_MUL_Q8A128_OP(__nv_bfloat16, silu_mul_q8a128_bf16)
