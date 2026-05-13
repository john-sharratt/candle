#pragma once
// FP8 E4M3 encode/decode utilities for quantization scale storage.
// E4M3: 1 sign + 4 exponent + 3 mantissa, exponent bias = 7
// Range: ±448, no infinity (exp=15 + man!=0 = NaN)

__device__ __forceinline__ float decode_e4m3(uint8_t val) {
    if (val == 0 || val == 0x80) return 0.0f;
    const int s = (val >> 7) & 1;
    const int e = (val >> 3) & 0xF;
    const int m = val & 0x7;
    float result;
    if (e == 0) {
        // Subnormal: 2^(1-7) * (m/8) = m * 2^(-9)
        result = scalbnf((float)m, -9);
    } else if (e == 15 && m != 0) {
        return 0.0f; // NaN → 0 for safety
    } else {
        // Normal: 2^(e-7) * (1 + m/8)
        result = scalbnf(1.0f + (float)m * 0.125f, e - 7);
    }
    return s ? -result : result;
}

__device__ __forceinline__ uint8_t encode_e4m3(float val) {
    const uint32_t bits = __float_as_uint(val);
    const uint32_t abs_bits = bits & 0x7fffffffu;

    // Zero (positive or negative)
    if (abs_bits == 0u) return 0u;

    const uint32_t sign = bits >> 31;

    // NaN → 0 for safety
    if (abs_bits > 0x7f800000u) return 0u;

    // Clamp: E4M3 max = 448.0 = 0x43E00000
    if (abs_bits >= 0x43E00000u) return (uint8_t)((sign << 7) | 0x77u);

    // fp32 unbiased exponent; 0 = subnormal fp32 (handled below via e8 <= 0)
    const int e8 = (int)(abs_bits >> 23) - 127 + 7;  // E4M3 biased exponent

    if (e8 <= 0) {
        // Subnormal E4M3: represented value = m * 2^(-9), m in [0,7]
        const int m = min(7, __float2int_rn(__uint_as_float(abs_bits) * 512.0f));
        return (uint8_t)((sign << 7) | (uint32_t)m);
    }

    // Normal: round fp32 23-bit mantissa to 3 bits (round-to-nearest-even via +0.5ulp)
    const uint32_t fp32_mant = abs_bits & 0x7fffffu;
    uint32_t m = (fp32_mant + (1u << 19)) >> 20;

    if (m >= 8u) {
        // Mantissa carry → increment exponent
        if (e8 + 1 >= 15) return (uint8_t)((sign << 7) | 0x77u);
        return (uint8_t)((sign << 7) | ((uint32_t)(e8 + 1) << 3));
    }
    return (uint8_t)((sign << 7) | ((uint32_t)e8 << 3) | m);
}

// Encode a non-negative scale value (no sign bit needed)
__device__ __forceinline__ uint8_t encode_e4m3_pos(float val) {
    return encode_e4m3(val);
}
