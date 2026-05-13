#pragma once

// =============================================================================
// CUDA HELPER FUNCTIONS FOR QUANTIZED KERNELS
// =============================================================================
// Low-level utility functions for working with quantized data formats.
//
// Key functions:
//   - get_int_from_uint8: Read 32-bit int from uint8 array (unaligned)
//   - get_int_from_uint8_aligned: Read 32-bit int from uint8 array (aligned)
//   - get_int_from_int8_aligned: Read 32-bit int from int8 array (aligned)
//   - ggml_cuda_dp4a: Byte-wise dot product (native __dp4a or fallback)
//
// =============================================================================

// Read 32-bit int from uint8 array with assumed 2-byte alignment
// Uses 2×uint16 reads to avoid unaligned 32-bit access
__device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32);
    
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    
    return x32;
}

// Read 32-bit int from uint8 array with strict 4-byte alignment
// Direct cast for maximum efficiency (assumes alignment maintained by caller)
__device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32));
}

// Read 32-bit int from int8 array with assumed 2-byte alignment
// Uses 2×uint16 reads to avoid unaligned 32-bit access
__device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32);
    
    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;
    
    return x32;
}

// Read 32-bit int from int8 array with strict 4-byte alignment
// Direct cast for maximum efficiency (assumes alignment maintained by caller)
__device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32));
}

// Byte-wise dot product: c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
// Uses native __dp4a intrinsic (available on SM61+, always present on SM80+)
__device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

// =============================================================================
// VEC_DOT IMPLEMENTATIONS FOR COMMON QUANTIZATION FORMATS
// =============================================================================
// These functions compute vec_dot(x, y) where:
// - x is quantized (Q4_0, Q4_1, etc.)
// - y is quantized as Q8_1
// Used by both batch matvec and tile-based implementations.

// Q4_0 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

// Q4_1 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16
    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

// Q5_0 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
    const float2 ds8f = __half22float2(ds8);
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

// Q5_1 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F;
        vi0    |= (vh[i] <<  4) & 0x00000010;
        vi0    |= (vh[i] << 11) & 0x00001000;
        vi0    |= (vh[i] << 18) & 0x00100000;
        vi0    |= (vh[i] << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F;
        vi1    |= (vh[i] >> 12) & 0x00000010;
        vi1    |= (vh[i] >>  5) & 0x00001000;
        vi1    |= (vh[i] <<  2) & 0x00100000;
        vi1    |= (vh[i] <<  9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }
#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16
    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

// Q8_0 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }
    return d8_0*d8_1 * sumi;
}

// Q8_1 + Q8_1 vec_dot
template <int vdr> __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }
#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16
    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

// =============================================================================
// K-QUANT VEC_DOT IMPLEMENTATIONS (Q2_K through Q6_K)
// =============================================================================

// Q2_K + Q8_1 MMVQ (contiguous v/x values)
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];
        const int vi = (v >> (2*i)) & 0x03030303;
        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
    }

    const float2 dm2f = __half22float2(dm2);
    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// Q2_K + Q8_1 MMQ (contiguous u/y values)
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;
        const int sc = scales[i0 / (QI8_1/2)];
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = ggml_cuda_dp4a(v[i], u[i], sumi_d_sc);
            sumi_m    = ggml_cuda_dp4a(m,    u[i], sumi_m);
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);
    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
}

// Q3_K + Q8_1 MMVQ (contiguous v/x values)
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;
        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;
        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2*i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = __vsubss4(vil, vih);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }

    return d3 * sumf;
}

// Q3_K + Q8_1 MMQ (contiguous u/y values)
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc);
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

// Q4_K + Q8_1 VMMQ (contiguous v/x values)
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// Q4_K + Q8_1 MMQ (contiguous u/y values)
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d);
        }

        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i];
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// Q5_K + Q8_1 VMMQ (contiguous v/x values)
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = __half22float2(dm5);
    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// Q5_K + Q8_1 MMQ (contiguous u/y values)
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d);
        }

        const float2 ds8f = __half22float2(ds8[i]);
        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i];
    }

    const float2 dm4f = __half22float2(dm4);
    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// Q6_K + Q8_1 MMVQ (contiguous v/x values)
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020);
        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
    }

    return d*sumf;
}

// Q6_K + Q8_1 MMQ (contiguous u/y values)
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}
