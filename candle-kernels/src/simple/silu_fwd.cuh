#pragma once
// The elementwise SiLU `usilu_*` computes for F32 and BF16 — one definition for
// that op and for every fusion that replaces it, so a fused SwiGLU is the same
// arithmetic as the eager `silu(gate) · up` rather than a second spelling of it.
//
// F32 is `fast_exp`'s; BF16 runs the generic `x / (1 + exp(-x))` in its own
// arithmetic, rounding at each step. `usilu_f16` is NOT this (it computes in F32
// and rounds once, `unary_utils.cuh`), so `__half` is deleted here rather than
// left to resolve to the generic form and quietly differ from the op.

#include "cuda_utils.cuh"
#include "../fast_exp.cuh"

template<typename T>
__device__ __forceinline__ T silu_fwd(T x) {
    return x / (static_cast<T>(1) + expg(-x));
}

template<>
__device__ __forceinline__ float silu_fwd<float>(float x) {
    return fast_exp::silu<float>(x);
}

template<>
__device__ __forceinline__ double silu_fwd<double>(double x) {
    return x / (1.0 + exp(-x));
}

template<>
__device__ __half silu_fwd<__half>(__half x) = delete;
