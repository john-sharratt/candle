#pragma once

#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include "cuda_fp8.h"

// Table showing which features are supported on which compute capability
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications

__device__ __forceinline__ __half atomicMaxf(__half* address, __half val) {
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmax_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMax is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMaxf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));        
    } else {
        return __int_as_float(atomicMax((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMaxf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMin((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMax((long long int *)addr, __double_as_longlong(value)));
    }
}


__device__ __forceinline__ __half atomicMinf(__half* address, __half val) {
    // Based on https://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
    unsigned short int* casted_address = (unsigned short int*)address;
    unsigned short int old = *casted_address;
    unsigned short int assumed;
    do {
        assumed = old;
        old = atomicCAS(casted_address, assumed, __half_as_ushort(__hmin_nan(val, __ushort_as_half(assumed))));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __ushort_as_half(old);
}

// atomicMin is not implemented for floats,
// solution copied https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
__device__ __forceinline__ float atomicMinf(float * addr, float value) {
    if (signbit(value)) {
        return __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
    } else {
        return __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    }
}

__device__ __forceinline__ double atomicMinf(double * addr, double value) {
    if (signbit(value)) {
        return __longlong_as_double(atomicMax((unsigned long long int *)addr, __double_as_longlong(value)));
    } else {
        return __longlong_as_double(atomicMin((long long int *)addr, __double_as_longlong(value)));
    }
}
