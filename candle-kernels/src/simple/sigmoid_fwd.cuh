#pragma once
// The elementwise sigmoid `usigmoid_*` computes for F32 and BF16 — one
// definition for that op and for every fusion that replaces it, so a fused
// kernel is the same arithmetic rather than a second spelling of it.
//
// F32 is `fast_exp`'s polynomial; BF16 runs the generic `1 / (1 + exp(-x))` in
// its own arithmetic, rounding at each step. `usigmoid_f16` is NOT this (it
// computes in F32 and rounds once, `unary_utils.cuh`), so `__half` is deleted
// here rather than left to resolve to the generic form and quietly differ.

#include "cuda_utils.cuh"
#include "../fast_exp.cuh"

template<typename T>
__device__ __forceinline__ T sigmoid_fwd(T x) {
    return recipg(static_cast<T>(1) + expg(-x));
}

template<>
__device__ __forceinline__ float sigmoid_fwd<float>(float x) {
    return fast_exp::sigmoid<float>(x);
}

template<>
__device__ __forceinline__ double sigmoid_fwd<double>(double x) {
    return 1.0 / (1.0 + exp(-x));
}

template<>
__device__ __half sigmoid_fwd<__half>(__half x) = delete;
