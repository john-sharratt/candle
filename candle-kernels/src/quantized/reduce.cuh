#pragma once

// =============================================================================
// WARP-LEVEL REDUCTION PRIMITIVES FOR QUANTIZED KERNELS
// =============================================================================
// Provides efficient warp-level reductions using butterfly shuffle pattern.
// All 32 lanes participate and receive the final result.
//
// Key functions:
//   - warp_reduce_sum_t<T>: Templated sum reduction
//   - warp_reduce_sum: Non-template float version
//   - warp_reduce_max: Max reduction for softmax
//
// Supported types: float, __half, __nv_bfloat16
// =============================================================================

// ============================================================================
// BF16 ARITHMETIC HELPER (needed for BF16 warp reduce)
// ============================================================================

/// BF16 addition — native __hadd on SM80+ (minimum supported architecture)
static __device__ __forceinline__ __nv_bfloat16 bf16_add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

// ============================================================================
// WARP-LEVEL SUM REDUCTION
// ============================================================================

/// Warp reduce using butterfly pattern (sum across all 32 lanes)
/// Input: value from current lane
/// Output: sum of all 32 input values (available to all lanes)
template <typename T>
static __device__ __forceinline__ T warp_reduce_sum_t(T x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

/// Specialization for __half - uses native __hadd for SM53+
/// Uses __half_as_ushort/__ushort_as_half to avoid address-of spills
template <>
__device__ __forceinline__ __half warp_reduce_sum_t<__half>(__half x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // Bit-cast in registers (no address-of, no spill)
        unsigned short y_bits = __shfl_xor_sync(0xffffffff, __half_as_ushort(x), mask, 32);
        x = __hadd(x, __ushort_as_half(y_bits));
    }
    return x;
}

/// Specialization for __half2 - reduces two halves in parallel
/// 2x throughput: one shuffle moves 32 bits (2 halves), one __hadd2 adds both
template <>
__device__ __forceinline__ __half2 warp_reduce_sum_t<__half2>(__half2 x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // Use union for register-safe bit casting
        union { __half2 h2; unsigned int u; } xu, yu;
        xu.h2 = x;
        yu.u = __shfl_xor_sync(0xffffffff, xu.u, mask, 32);
        x = __hadd2(x, yu.h2);
    }
    return x;
}

/// Specialization for __nv_bfloat16 - uses raw bit shuffle + bf16_add
/// Uses __bfloat16_as_ushort/__ushort_as_bfloat16 to avoid address-of spills
template <>
__device__ __forceinline__ __nv_bfloat16 warp_reduce_sum_t<__nv_bfloat16>(__nv_bfloat16 x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // Bit-cast in registers (no address-of, no spill)
        unsigned short y_bits = __shfl_xor_sync(0xffffffff, __bfloat16_as_ushort(x), mask, 32);
        x = bf16_add(x, __ushort_as_bfloat16(y_bits));
    }
    return x;
}

/// Specialization for __nv_bfloat162 - reduces two bf16s in parallel
/// 2x throughput: one shuffle moves 32 bits (2 bf16s), one __hadd2 adds both
template <>
__device__ __forceinline__ __nv_bfloat162 warp_reduce_sum_t<__nv_bfloat162>(__nv_bfloat162 x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        // Use union for register-safe bit casting
        union { __nv_bfloat162 b2; unsigned int u; } xu, yu;
        xu.b2 = x;
        yu.u = __shfl_xor_sync(0xffffffff, xu.u, mask, 32);
        x = __hadd2(x, yu.b2);
    }
    return x;
}

/// Non-template float version for compatibility
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    return warp_reduce_sum_t<float>(x);
}

// ============================================================================
// WARP-LEVEL MAX REDUCTION
// ============================================================================

/// Warp reduce max using butterfly pattern (for softmax, etc.)
/// Input: value from current lane
/// Output: max of all 32 input values (available to all lanes)
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}
