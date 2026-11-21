// The operand packing and the MMA wrappers for SM80+ and the FP8 variant for
// SM89+.
//
// `mma_sync_m16n8k16_row_col_f32` is the float-accumulating MMA for SM80+, and
// `mma_sync_m16n8k32_row_col_f32_fp8` is the FP8 E4M3 variant for SM89+.
// `__pack_half2`, `pack_fp8x4` and `cvt_f16x4_to_fp8x4` pack operands for them,
// and `unpack_fp8x4` converts packed FP8 back to floats.

// ============================================================================
// MMA (MATRIX MULTIPLY-ACCUMULATE) SUPPORT FOR SM80+
// ============================================================================

/// Pack two 16-bit values into uint32 for MMA operand
template <typename T>
__device__ __forceinline__ uint32_t __pack_half2(T a, T b);

template <>
__device__ __forceinline__ uint32_t __pack_half2<__half>(__half a, __half b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__half_as_ushort(a)),
          "h"(__half_as_ushort(b)));
    return result;
}

template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__bfloat16_as_ushort(a)),
          "h"(__bfloat16_as_ushort(b)));
    return result;
}

// Float pack - convert to half for MMA
template <>
__device__ __forceinline__ uint32_t __pack_half2<float>(float a, float b) {
    __half ha = __float2half_rn(a);
    __half hb = __float2half_rn(b);
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(__half_as_ushort(ha)),
          "h"(__half_as_ushort(hb)));
    return result;
}

// FP8 pack - for m16n8k16 MMA we convert to half first (m16n8k32 uses native FP8)
template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t sa = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
    __nv_fp8_storage_t sb = *reinterpret_cast<const __nv_fp8_storage_t*>(&b);
    __half ha = __nv_cvt_fp8_to_halfraw(sa, __NV_E4M3);
    __half hb = __nv_cvt_fp8_to_halfraw(sb, __NV_E4M3);
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&ha)),
          "h"(*reinterpret_cast<const unsigned short*>(&hb)));
    return result;
#else
    // Fallback: convert to float, then pack as float
    return __pack_half2<float>((float)a, (float)b);
#endif
}

/// m16n8k16 MMA with float32 accumulation (SM80+)
/// A operand: 16×16 (4 uint32 registers, each holding 4 F16 values)
/// B operand: 16×8  (2 uint32 registers, each holding 4 F16 values)
/// C output:  16×8  (4 float registers, accumulated)
template <typename T>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    } else {
        // Default to F16 MMA for half or converted FP8
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
}

/// m16n8k32 MMA with FP8 E4M3 inputs and float32 accumulation (SM89+)
/// A operand: 16×32 (4 uint32 registers, each holding 4 FP8 values)
/// B operand: 32×8  (2 uint32 registers, each holding 4 FP8 values)
/// C output:  16×8  (4 float registers, accumulated) - SAME as F16!
///
/// Note: Processes K=32 per instruction (2× throughput vs F16's K=16)
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f32_fp8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    // Pre-SM89: no FP8 MMA support
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

/// Pack 4 consecutive FP8 values from memory into uint32 for MMA operand
/// Simply reinterprets 4 contiguous FP8 bytes as a uint32
__device__ __forceinline__ uint32_t pack_fp8x4(const __nv_fp8_e4m3* ptr) {
    return *reinterpret_cast<const uint32_t*>(ptr);
}

/// Convert 4 F16 values (as 2×half2) → packed FP8x4 uint32
/// Used for converting dequantized weights to FP8 for m16n8k32 MMA
__device__ __forceinline__ uint32_t cvt_f16x4_to_fp8x4(__half2 h01, __half2 h23) {
    __half h0 = __low2half(h01);
    __half h1 = __high2half(h01);
    __half h2 = __low2half(h23);
    __half h3 = __high2half(h23);
    uint32_t result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h0), __NV_SATFINITE, __NV_E4M3);
    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h1), __NV_SATFINITE, __NV_E4M3);
    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h2), __NV_SATFINITE, __NV_E4M3);
    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h3), __NV_SATFINITE, __NV_E4M3);
#else
    // Fallback: use bit-casting for SM80-SM88 (limited FP8 support)
    // Pack 4 __half values using bitwise operations
    uint16_t* shorts = reinterpret_cast<uint16_t*>(&result);
    shorts[0] = *reinterpret_cast<uint16_t*>(&h0);
    shorts[1] = *reinterpret_cast<uint16_t*>(&h1);
#endif
    return result;
}

// ============================================================================
// UNPACK_FP8X4 - Convert 4 packed FP8 bytes to 4 floats
// ============================================================================
// CRITICAL: Must use __nv_cvt_fp8_to_halfraw to interpret raw bits as FP8,
// NOT __nv_fp8_e4m3(storage_t) which treats the value as a number to convert.
__device__ __forceinline__ void unpack_fp8x4(uint32_t packed, float& f0, float& f1, float& f2, float& f3) {
    f0 = __half2float(__nv_cvt_fp8_to_halfraw(packed & 0xFF, __NV_E4M3));
    f1 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 8) & 0xFF, __NV_E4M3));
    f2 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 16) & 0xFF, __NV_E4M3));
    f3 = __half2float(__nv_cvt_fp8_to_halfraw((packed >> 24) & 0xFF, __NV_E4M3));
}
