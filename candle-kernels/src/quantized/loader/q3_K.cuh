#pragma once

// =============================================================================
// Q3_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 3-bit weights (2-bit low + 1-bit high)
// - qs: 16 bits = 8 × 2-bit low crumbs
// - qh: 8 bits = 8 × 1-bit high bits
//
// Q3_K formula: q3 = ql + 4*qh - 4 (signed range: -4 to +3)
//               dequant = d * q3
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q3_K_inline {
    using acc_type = acc_t;
    
    uint16_t qs;             // 8 × 2-bit low crumbs
    uint8_t qh;              // 8 × 1-bit high bits
    acc_t scale;             // d scale
    
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;
    }
    
    // Compute signed 3-bit value: ql + 4*qh - 4
    // Range: ql=0-3, qh=0-1 → result = -4 to +3
    __device__ __forceinline__ static int get_q3(int ql, int qh_bit) {
        return ql + (qh_bit << 2) - 4;
    }
    
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q3_K* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q3_K uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q3_K_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q3_K_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Q3_K K/128 layout (96 bytes, 8 groups of 12 bytes each):
        // Group g (threads 2g, 2g+1) at byte offset g*12:
        //   offset 0-1: qs_even (2B) - thread 2g low bits
        //   offset 2:   qh_even (1B) - thread 2g high bits  
        //   offset 3:   padding (1B)
        //   offset 4-7: dm (4B) - half2 scale for both threads
        //   offset 8:   qh_odd (1B) - thread 2g+1 high bits
        //   offset 9:   padding (1B)
        //   offset 10-11: qs_odd (2B) - thread 2g+1 low bits
        
        const int group = lane >> 1;        // which thread pair (0-7)
        const int in_group = lane & 1;      // 0 or 1 within pair
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(blk);
        const int base = group * 12;
        
        if (in_group == 0) {
            // Even thread: qs at offset 0, qh at offset 2
            qs = *reinterpret_cast<const uint16_t*>(&bytes[base + 0]);
            qh = bytes[base + 2];
        } else {
            // Odd thread: qs at offset 10, qh at offset 8
            qs = *reinterpret_cast<const uint16_t*>(&bytes[base + 10]);
            qh = bytes[base + 8];
        }
        
        // dm is at offset 4 within each 12-byte group
        const half* d_ptr = reinterpret_cast<const half*>(&bytes[base + 4]);
        scale = to_acc<acc_t>(__half2float(*d_ptr));
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q3_K uses 16-thread interface");
        
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        if constexpr (std::is_same_v<y_t, float>) {
            const float d = to_f32(scale);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            float sum = 0.0f;
            // Elements 0-3
            {
                const float4 yv = y4[0];
                sum = __fmaf_rn(d * float(get_q3((q >> 0) & 0x3, (h >> 0) & 1)), yv.x, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 2) & 0x3, (h >> 1) & 1)), yv.y, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 4) & 0x3, (h >> 2) & 1)), yv.z, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 6) & 0x3, (h >> 3) & 1)), yv.w, sum);
            }
            // Elements 4-7
            {
                const float4 yv = y4[1];
                sum = __fmaf_rn(d * float(get_q3((q >> 8) & 0x3, (h >> 4) & 1)), yv.x, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 10) & 0x3, (h >> 5) & 1)), yv.y, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 12) & 0x3, (h >> 6) & 1)), yv.z, sum);
                sum = __fmaf_rn(d * float(get_q3((q >> 14) & 0x3, (h >> 7) & 1)), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + PRMT OPTIMIZATION for signed Q3_K
            // Q3_K: q3 = ql + 4*qh - 4 (signed range: -4 to +3)
            // We compute unsigned (ql + 4*qh) first, then subtract bias 4
            const half d_h = __float2half(to_f32(scale));
            const half2 d2 = __half2half2(d_h);
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // Extract crumbs (2-bit values) from qs (16 bits = 8 × 2-bit)
            // qs layout: q0 at bits 0-1, q1 at bits 2-3, ..., q7 at bits 14-15
            // We need to place them in bytes: byte0=q0, byte1=q2, byte2=q4, byte3=q6 (even)
            //                                 byte0=q1, byte1=q3, byte2=q5, byte3=q7 (odd)
            const uint32_t ql_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t ql_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            
            // Extract high bits from qh (8 bits) and place in bit 2 of each byte
            const uint32_t hb_even = ((h & 0x01) << 2) |
                                     (((h >> 2) & 0x01) << 10) |
                                     (((h >> 4) & 0x01) << 18) |
                                     (((h >> 6) & 0x01) << 26);
            const uint32_t hb_odd = (((h >> 1) & 0x01) << 2) |
                                    (((h >> 3) & 0x01) << 10) |
                                    (((h >> 5) & 0x01) << 18) |
                                    (((h >> 7) & 0x01) << 26);
            
            // Combine: unsigned q3 = ql + 4*qh (range 0-7)
            const uint32_t q3_even = ql_even | hb_even;
            const uint32_t q3_odd = ql_odd | hb_odd;
            
            // LOP3 magic constants for FP16 (unsigned 0-7)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x00070007;  // 3-bit mask
            
            // Signed bias: we need to subtract 4 from each value
            // In FP16, 4.0 = 0x4400
            constexpr uint32_t SIGNED_BIAS = 0x44004400;
            
            // STREAMING + PRMT OPTIMIZATION
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y2[0], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y2[1], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y2[2], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 + PRMT OPTIMIZATION for signed Q3_K
            const __nv_bfloat16 d_bf = __float2bfloat16(to_f32(scale));
            const __nv_bfloat162 d2 = __bfloat162bfloat162(d_bf);
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // Extract crumbs and high bits (same as FP16 path)
            const uint32_t ql_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t ql_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            const uint32_t hb_even = ((h & 0x01) << 2) |
                                     (((h >> 2) & 0x01) << 10) |
                                     (((h >> 4) & 0x01) << 18) |
                                     (((h >> 6) & 0x01) << 26);
            const uint32_t hb_odd = (((h >> 1) & 0x01) << 2) |
                                    (((h >> 3) & 0x01) << 10) |
                                    (((h >> 5) & 0x01) << 18) |
                                    (((h >> 7) & 0x01) << 26);
            const uint32_t q3_even = ql_even | hb_even;
            const uint32_t q3_odd = ql_odd | hb_odd;
            
            // LOP3 magic constants for BF16
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK = 0x00070007;
            // BF16 4.0 = 0x4080
            constexpr uint32_t SIGNED_BIAS_BF = 0x40804080;
            
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w_unsigned = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                                    *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                __nv_bfloat162 w = __hsub2(w_unsigned, *reinterpret_cast<const __nv_bfloat162*>(&SIGNED_BIAS_BF));
                sum2 = __hfma2(__hmul2(d2, w), y2[0], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w_unsigned = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                                    *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                __nv_bfloat162 w = __hsub2(w_unsigned, *reinterpret_cast<const __nv_bfloat162*>(&SIGNED_BIAS_BF));
                sum2 = __hfma2(__hmul2(d2, w), y2[1], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w_unsigned = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                                    *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                __nv_bfloat162 w = __hsub2(w_unsigned, *reinterpret_cast<const __nv_bfloat162*>(&SIGNED_BIAS_BF));
                sum2 = __hfma2(__hmul2(d2, w), y2[2], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w_unsigned = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                                    *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                __nv_bfloat162 w = __hsub2(w_unsigned, *reinterpret_cast<const __nv_bfloat162*>(&SIGNED_BIAS_BF));
                sum2 = __hfma2(__hmul2(d2, w), y2[3], sum2);
            }
            
            return __bfloat162float(__low2bfloat16(sum2)) + __bfloat162float(__high2bfloat16(sum2));
            
        } else {
            // =================================================================
            // FP8 SPECIALIZED PATH - High performance via FP16 accumulation
            // =================================================================
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half d_h = __float2half(to_f32(scale));
            const half2 d2 = __half2half2(d_h);
            
            // Load FP8 inputs as uint32_t
            const uint32_t* y_u32 = reinterpret_cast<const uint32_t*>(y + get_lane() * 8);
            const uint32_t y_packed0 = y_u32[0];
            const uint32_t y_packed1 = y_u32[1];
            
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            // Hardware FP8→FP16 conversion for SM89+
            const __nv_fp8x2_storage_t* fp8_ptr0 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed0);
            const __nv_fp8x2_storage_t* fp8_ptr1 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed1);
            
            __half2_raw y0_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[0], __NV_E4M3);
            __half2_raw y1_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[1], __NV_E4M3);
            __half2_raw y2_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[0], __NV_E4M3);
            __half2_raw y3_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[1], __NV_E4M3);
            
            const half2 y0 = *reinterpret_cast<half2*>(&y0_raw);
            const half2 y1 = *reinterpret_cast<half2*>(&y1_raw);
            const half2 y2_v = *reinterpret_cast<half2*>(&y2_raw);
            const half2 y3 = *reinterpret_cast<half2*>(&y3_raw);
            #else
            // Software conversion for SM80-SM88: use precomputed 256-entry LUT
            half2 y0, y1, y2_v, y3;
            fp8x4_to_half2x2_lut(y_packed0, y0, y1);
            fp8x4_to_half2x2_lut(y_packed1, y2_v, y3);
            #endif
            
            // Extract crumbs and high bits
            const uint32_t ql_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t ql_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            const uint32_t hb_even = ((h & 0x01) << 2) |
                                     (((h >> 2) & 0x01) << 10) |
                                     (((h >> 4) & 0x01) << 18) |
                                     (((h >> 6) & 0x01) << 26);
            const uint32_t hb_odd = (((h >> 1) & 0x01) << 2) |
                                    (((h >> 3) & 0x01) << 10) |
                                    (((h >> 5) & 0x01) << 18) |
                                    (((h >> 7) & 0x01) << 26);
            const uint32_t q3_even = ql_even | hb_even;
            const uint32_t q3_odd = ql_odd | hb_odd;
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x00070007;
            constexpr uint32_t SIGNED_BIAS = 0x44004400;  // FP16 4.0
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y0, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y1, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y2_v, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q3_even, q3_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w_unsigned = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w = __hsub2(w_unsigned, *reinterpret_cast<const half2*>(&SIGNED_BIAS));
                sum2 = __hfma2(__hmul2(d2, w), y3, sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N < 16, "Q3_K uses 16-thread interface");
        
        const float d = to_f32(scale);
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        out4[0] = make_float4(
            d * float(get_q3((q >> 0) & 0x3, (h >> 0) & 1)),
            d * float(get_q3((q >> 2) & 0x3, (h >> 1) & 1)),
            d * float(get_q3((q >> 4) & 0x3, (h >> 2) & 1)),
            d * float(get_q3((q >> 6) & 0x3, (h >> 3) & 1))
        );
        out4[1] = make_float4(
            d * float(get_q3((q >> 8) & 0x3, (h >> 4) & 1)),
            d * float(get_q3((q >> 10) & 0x3, (h >> 5) & 1)),
            d * float(get_q3((q >> 12) & 0x3, (h >> 6) & 1)),
            d * float(get_q3((q >> 14) & 0x3, (h >> 7) & 1))
        );
    }
};

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q3_K, vdr, act_t> {
    using type = vec_dot_q_loader_q3_K_inline<vdr, acc_for_act_t<act_t>>;
};

// K/128 compact format uses the same inline loader
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q3_K, vdr, act_t> {
    using type = vec_dot_q_loader_q3_K_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// SCALE EXTRACTION FOR Q3_K
// =============================================================================
//
// Q3_K K/128 format: 8 thread pairs, each pair shares a dm (half2).
// 12 bytes per pair: [qs_even(2), qh_even(1), pad(1), dm(4), qh_odd(1), qs_odd(2), pad(1)]
// Total: 8 pairs × 12B = 96B per K/128 block
// Output: 8 scales per K/128 block (1 scale per 16 elements)
//
// =============================================================================

namespace gemx_q3_K {

// Extract all scales: row-major [N, K/128] → column-major [K/16, N]
template <typename ScaleT>
__device__ inline void extract_scales_impl(
    const block_q3_K* __restrict__ x,
    ScaleT* __restrict__ scales_out,
    int nrows,
    int ncols
) {
    constexpr int ELEMENTS_PER_BLOCK = 128;
    constexpr int SCALES_PER_BLOCK = 8;  // 8 thread pairs, 1 scale per 16 elements
    constexpr int ELEMENTS_PER_SCALE = 16;
    constexpr int BYTES_PER_GROUP = 12;
    
    const int blocks_per_row = ncols / ELEMENTS_PER_BLOCK;
    const int scales_per_row = ncols / ELEMENTS_PER_SCALE;
    const int total_scales = nrows * scales_per_row;
    
    for (int src_scale_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         src_scale_idx < total_scales; 
         src_scale_idx += blockDim.x * gridDim.x) 
    {
        const int row = src_scale_idx / scales_per_row;
        const int scale_col = src_scale_idx % scales_per_row;
        const int block_col = scale_col / SCALES_PER_BLOCK;
        const int local_scale = scale_col % SCALES_PER_BLOCK;  // 0-7 (pair index)
        const int block_idx = row * blocks_per_row + block_col;
        
        // Reinterpret as K/128 block
        const block_c_q3_K_k128* blk = 
            reinterpret_cast<const block_c_q3_K_k128*>(&x[block_idx]);
        
        // Each pair's dm is at byte offset: group * 12 + 4
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(blk);
        const int dm_byte_offset = local_scale * BYTES_PER_GROUP + 4;
        const half2 dm = *reinterpret_cast<const half2*>(bytes + dm_byte_offset);
        
        // Output scale: just d (the scale factor)
        const int dst_scale_idx = scale_col * nrows + row;
        scales_out[dst_scale_idx] = __low2half(dm);
    }
}

} // namespace gemx_q3_K

// =============================================================================
// GEMX DEQUANT TRAITS - Q3_K (3-bit K-quant, symmetric signed)
// =============================================================================
// Q3_K formula: q3 = ql + 4*qh - 4 (signed range: -4 to +3)
//   ql: 2-bit low crumbs (0-3)
//   qh: 1-bit high bits (0-1)
//   unsigned: ql + 4*qh = 0-7
//   signed: subtract 4 → -4 to +3
//
// Include the base gemx_dequant infrastructure
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q3_K, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = false;  // Q3_K is symmetric (signed, no min)
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q3_K>::scales_per_ktile;
    static constexpr int bits_per_element = 3;
    
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // MMA LANE-DISPATCHED DEQUANTIZATION - FULLY COMPILE-TIME
    // =========================================================================
    //
    // Q3_K K/128 layout (96 bytes):
    //   8 groups of 12 bytes: [qs_even(2) + qh_even(1) + dm(4) + qh_odd(1) + qs_odd(2) + pad(2)]
    //   Thread 2g at group g: qs at offset g*12+0, qh at offset g*12+2
    //   Thread 2g+1 at group g: qs at offset g*12+9, qh at offset g*12+7
    //   dm at offset g*12+3
    //
    // MMA m16n8k16: 32 threads → 8 rows × 4 K-groups
    //   Lane/4 = N (row 0-7), Lane%4 = K_GROUP (0-3)
    //   Each thread provides 4 elements for K=16 MMA
    //
    // =========================================================================
    
    static constexpr int K128_BYTES = 96;
    
    // Q3_K byte offsets for each thread (0-15) within K/128 block
    // qs is 2 bytes (uint16_t), qh is 1 byte (uint8_t)
    static constexpr int qs_byte_offset[16] = {
        0, 9, 12, 21, 24, 33, 36, 45,   // threads 0-7
        48, 57, 60, 69, 72, 81, 84, 93  // threads 8-15
    };
    static constexpr int qh_byte_offset[16] = {
        2, 7, 14, 19, 26, 31, 38, 43,   // threads 0-7
        50, 55, 62, 67, 74, 79, 86, 91  // threads 8-15
    };
    // dm byte offsets: threads 0-1 share dm0, 2-3 share dm1, etc.
    static constexpr int dm_byte_offset[8] = {3, 15, 27, 39, 51, 63, 75, 87};
    
    // -------------------------------------------------------------------------
    // COMPILE-TIME LANE PARAMETERS
    // -------------------------------------------------------------------------
    
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;           // Row index (0-7)
        static constexpr int K_GROUP = LANE % 4;     // K-group within K=16 (0-3)
        
        // For K=16: each iteration covers 16 elements
        // K_GROUP 0: elements 0-3, K_GROUP 1: elements 4-7, etc.
        // Element i → thread (i/8), nibble (i%8)
        static constexpr int THREAD_IN_BLOCK = K_GROUP / 2;
        static constexpr int NIBBLE_HALF = K_GROUP % 2;  // 0=first 4 elem, 1=second 4 elem
    };
    
    // -------------------------------------------------------------------------
    // BUILD Q3 ARRAYS: Combine ql (2-bit) and qh (1-bit) into 3-bit values
    // -------------------------------------------------------------------------
    
    __device__ __forceinline__ static void build_q3_arrays(
        uint16_t qs, uint8_t qh,
        uint32_t& q3_even, uint32_t& q3_odd
    ) {
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        const uint32_t ql_even = ((q >> 0) & 0x3) |
                                 (((q >> 4) & 0x3) << 8) |
                                 (((q >> 8) & 0x3) << 16) |
                                 (((q >> 12) & 0x3) << 24);
        const uint32_t ql_odd = ((q >> 2) & 0x3) |
                                (((q >> 6) & 0x3) << 8) |
                                (((q >> 10) & 0x3) << 16) |
                                (((q >> 14) & 0x3) << 24);
        
        const uint32_t hb_even = ((h & 0x01) << 2) |
                                 (((h >> 2) & 0x01) << 10) |
                                 (((h >> 4) & 0x01) << 18) |
                                 (((h >> 6) & 0x01) << 26);
        const uint32_t hb_odd = (((h >> 1) & 0x01) << 2) |
                                (((h >> 3) & 0x01) << 10) |
                                (((h >> 5) & 0x01) << 18) |
                                (((h >> 7) & 0x01) << 26);
        
        q3_even = ql_even | hb_even;
        q3_odd = ql_odd | hb_odd;
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS for K=16 MMA with compile-time NIBBLE_HALF
    // -------------------------------------------------------------------------
    
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(
        uint32_t q3_even, uint32_t q3_odd, FragB_t& frag
    ) {
        constexpr int LO_MASK = 0x00070007;  // 3-bit mask
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // NIBBLE_HALF=0: pairs 0,1 (elements 0-3)
            // NIBBLE_HALF=1: pairs 2,3 (elements 4-7)
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q3_even, q3_odd) : 
                prmt_build_lop3_pair_2(q3_even, q3_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q3_even, q3_odd) : 
                prmt_build_lop3_pair_3(q3_even, q3_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q3_even, q3_odd) : 
                prmt_build_lop3_pair_2(q3_even, q3_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q3_even, q3_odd) : 
                prmt_build_lop3_pair_3(q3_even, q3_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q3_even, q3_odd) : 
                prmt_build_lop3_pair_2(q3_even, q3_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q3_even, q3_odd) : 
                prmt_build_lop3_pair_3(q3_even, q3_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX_FP16);
            
            half2 h01 = __hsub2(*reinterpret_cast<half2*>(&w01),
                                *reinterpret_cast<const half2*>(&SUB_FP16));
            half2 h23 = __hsub2(*reinterpret_cast<half2*>(&w23),
                                *reinterpret_cast<const half2*>(&SUB_FP16));
            
            uint16_t fp8_01 = __nv_cvt_halfraw2_to_fp8x2(
                *reinterpret_cast<__half2_raw*>(&h01), __NV_SATFINITE, __NV_E4M3);
            uint16_t fp8_23 = __nv_cvt_halfraw2_to_fp8x2(
                *reinterpret_cast<__half2_raw*>(&h23), __NV_SATFINITE, __NV_E4M3);
            
            frag[0] = (static_cast<uint32_t>(fp8_23) << 16) | fp8_01;
        }
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 8 ELEMENTS for K=32 MMA (FP8)
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(
        uint32_t q3_even, uint32_t q3_odd, FragB_t& frag0, FragB_t& frag1
    ) {
        extract_4_elements<0>(q3_even, q3_odd, frag0);
        extract_4_elements<1>(q3_even, q3_odd, frag1);
    }
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // MMA m16n8k16 requires:
    //   frag[0] = half2(B[k0, n], B[k1, n]) where k0=(lane%4)*2, k1=k0+1
    //   frag[1] = half2(B[k0+8, n], B[k1+8, n])
    //
    // Q3_K K/128 layout: 16 threads × 8 elements (3-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7 (qs=2B low bits, qh=1B high bits)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo/qh_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi/qh_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //   dm = shared scale for this K/16 slice
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // ADDRESS COMPUTATION
        // =====================================================================
        const int row = lane >> 2;                  // N: 0-7
        const int k_group = lane & 3;               // K_GROUP: 0-3
        
        // Base pointer for this row (Q3_K: 96 bytes per row)
        const uint8_t* row_bytes = smem_rows + row * K128_BYTES;
        
        // Q3_K K/128 byte layout (12 bytes per thread pair):
        // Group g at offset g*12: qs0(2B), qh0(1B), pad(1B), dm(4B), qh1(1B), pad(1B), qs1(2B)
        //
        // Thread indices for k_iter:
        //   thread_lo = k_iter*2 (elements 0-7)
        //   thread_hi = k_iter*2+1 (elements 8-15)
        //
        // Byte offsets:
        static constexpr int local_qs_byte_offset[16] = {
            0, 10, 12, 22, 24, 34, 36, 46,    // threads 0-7
            48, 58, 60, 70, 72, 82, 84, 94   // threads 8-15
        };
        static constexpr int local_qh_byte_offset[16] = {
            2, 8, 14, 20, 26, 32, 38, 44,     // threads 0-7
            50, 56, 62, 68, 74, 80, 86, 92   // threads 8-15
        };
        static constexpr int local_dm_byte_offset[8] = {4, 16, 28, 40, 52, 64, 76, 88};
        
        const int thread_lo = k_iter << 1;      // thread for elements 0-7
        const int thread_hi = thread_lo + 1;    // thread for elements 8-15
        
        // Load qs (2B) and qh (1B) for both element ranges
        const uint16_t qs_lo = *reinterpret_cast<const uint16_t*>(row_bytes + local_qs_byte_offset[thread_lo]);
        const uint16_t qs_hi = *reinterpret_cast<const uint16_t*>(row_bytes + local_qs_byte_offset[thread_hi]);
        const uint8_t qh_lo = *(row_bytes + local_qh_byte_offset[thread_lo]);
        const uint8_t qh_hi = *(row_bytes + local_qh_byte_offset[thread_hi]);
        const half2 dm = *reinterpret_cast<const half2*>(row_bytes + local_dm_byte_offset[k_iter]);
        
        // Q3_K scale: d is in low half of dm (dm.x), m is not used (symmetric)
        const half d_h = __low2half(dm);
        
        // =====================================================================
        // ELEMENT EXTRACTION
        // =====================================================================
        // MMA k_group determines which K positions:
        //   k_group=0: K = 0,1,8,9
        //   k_group=1: K = 2,3,10,11
        //   k_group=2: K = 4,5,12,13
        //   k_group=3: K = 6,7,14,15
        //
        // From qs_lo/qh_lo (elements 0-7): extract element pair at k_group*2, k_group*2+1
        // From qs_hi/qh_hi (elements 8-15): same indices relative to that thread
        
        const int shift = k_group * 4;      // 2 bits per element, 2 elements = 4 bits shift
        const int h_shift = k_group * 2;    // 1 bit per element, 2 elements = 2 bits shift
        
        // Extract q3 values: q3[i] = ql[i] | (qh[i] << 2)
        // For frag[0]: elements from qs_lo/qh_lo (K positions 0-7)
        const uint32_t q3_lo_0 = ((qs_lo >> shift) & 0x3) | (((qh_lo >> h_shift) & 1) << 2);
        const uint32_t q3_lo_1 = ((qs_lo >> (shift + 2)) & 0x3) | (((qh_lo >> (h_shift + 1)) & 1) << 2);
        const uint32_t q_lo_pair = q3_lo_0 | (q3_lo_1 << 16);
        
        // For frag[1]: elements from qs_hi/qh_hi (K positions 8-15)
        const uint32_t q3_hi_0 = ((qs_hi >> shift) & 0x3) | (((qh_hi >> h_shift) & 1) << 2);
        const uint32_t q3_hi_1 = ((qs_hi >> (shift + 2)) & 0x3) | (((qh_hi >> (h_shift + 1)) & 1) << 2);
        const uint32_t q_hi_pair = q3_hi_0 | (q3_hi_1 << 16);
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q3_K: w = d * (q3 - 4) (symmetric quantization)
        // =====================================================================
        constexpr int LO_MASK = 0x00070007;  // 3-bit mask
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64046404;  // Subtract 1028.0 (1024+4) in fp16
            const half2 scale2 = __half2half2(d_h);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43044304;  // Subtract 132.0 (128+4) in bf16
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_h)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64046404;
            const half2 scale2 = __half2half2(d_h);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            half2 w0 = __hmul2(scale2, raw0);
            half2 w1 = __hmul2(scale2, raw1);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY OPTIMIZED)
    // =========================================================================
    // Optimizations applied:
    // 1. Vector loads: 3× int4 for all 48 bytes (4 groups × 12B)
    // 2. Single shift computation for k_group, amortized across all slices  
    // 3. Inline q3 extraction (pure ALU, no function calls)
    // 4. No loop/function call overhead - fully unrolled
    //
    // Q3_K K/128 layout (96 bytes = 8 groups × 12B):
    //   Each 12B group: [qs_even:2][qh_even:1][pad:1][dm:4][qh_odd:1][pad:1][qs_odd:2]
    //   half_idx=0: groups 0-3 (bytes 0-47)
    //   half_idx=1: groups 4-7 (bytes 48-95)
    // =========================================================================
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_bytes = smem_rows + row * K128_BYTES;
        
        // Shift for k_group: extracts the right 2-bit pair from qs
        const int shift = k_group * 4;      // 0, 4, 8, 12
        const int h_shift = k_group * 2;    // 0, 2, 4, 6
        
        // Compile-time base offset for this half
        constexpr int base = (half_idx == 0) ? 0 : 48;
        
        // Load all 48 bytes (4 groups × 12B) as 3× int4
        // Bytes 0-15, 16-31, 32-47 relative to base
        const int4 d0 = *reinterpret_cast<const int4*>(row_bytes + base + 0);
        const int4 d1 = *reinterpret_cast<const int4*>(row_bytes + base + 16);
        const int4 d2 = *reinterpret_cast<const int4*>(row_bytes + base + 32);
        
        // Extract qs, qh, dm from the loaded data
        // Group layout: [qs_even:2 + qh_even:1 + pad:1][dm:4][qh_odd:1 + pad:1 + qs_odd:2]
        //
        // d0 covers bytes 0-15:  group0 (0-11) + group1 bytes 0-3
        // d1 covers bytes 16-31: group1 bytes 4-11 + group2 bytes 0-7
        // d2 covers bytes 32-47: group2 bytes 8-11 + group3 (0-11)
        
        // Group 0 (bytes 0-11): d0.x=bytes 0-3, d0.y=bytes 4-7, d0.z=bytes 8-11
        const uint32_t g0_word0 = d0.x;  // qs_even[0:15], qh_even[16:23], pad[24:31]
        const uint32_t g0_dm = d0.y;     // dm as uint32
        const uint32_t g0_word2 = d0.z;  // qh_odd[0:7], pad[8:15], qs_odd[16:31]
        
        // Group 1 (bytes 12-23): d0.w=bytes 12-15, d1.x=bytes 16-19, d1.y=bytes 20-23
        const uint32_t g1_word0 = d0.w;
        const uint32_t g1_dm = d1.x;
        const uint32_t g1_word2 = d1.y;
        
        // Group 2 (bytes 24-35): d1.z=bytes 24-27, d1.w=bytes 28-31, d2.x=bytes 32-35
        const uint32_t g2_word0 = d1.z;
        const uint32_t g2_dm = d1.w;
        const uint32_t g2_word2 = d2.x;
        
        // Group 3 (bytes 36-47): d2.y=bytes 36-39, d2.z=bytes 40-43, d2.w=bytes 44-47
        const uint32_t g3_word0 = d2.y;
        const uint32_t g3_dm = d2.z;
        const uint32_t g3_word2 = d2.w;
        
        // Extract qs and qh from packed words
        // word0: qs_even at bits 0-15, qh_even at bits 16-23
        // word2: qh_odd at bits 0-7, qs_odd at bits 16-31
        #define EXTRACT_QS_QH(word0, word2, qs_lo, qs_hi, qh_lo, qh_hi) \
            const uint16_t qs_lo = (word0) & 0xFFFF; \
            const uint8_t qh_lo = ((word0) >> 16) & 0xFF; \
            const uint16_t qs_hi = ((word2) >> 16) & 0xFFFF; \
            const uint8_t qh_hi = (word2) & 0xFF
        
        EXTRACT_QS_QH(g0_word0, g0_word2, qs0_lo, qs0_hi, qh0_lo, qh0_hi);
        EXTRACT_QS_QH(g1_word0, g1_word2, qs1_lo, qs1_hi, qh1_lo, qh1_hi);
        EXTRACT_QS_QH(g2_word0, g2_word2, qs2_lo, qs2_hi, qh2_lo, qh2_hi);
        EXTRACT_QS_QH(g3_word0, g3_word2, qs3_lo, qs3_hi, qh3_lo, qh3_hi);
        
        #undef EXTRACT_QS_QH
        
        // Extract dm (scale) as half from each group
        const half dm0 = *reinterpret_cast<const half*>(&g0_dm);
        const half dm1 = *reinterpret_cast<const half*>(&g1_dm);
        const half dm2 = *reinterpret_cast<const half*>(&g2_dm);
        const half dm3 = *reinterpret_cast<const half*>(&g3_dm);
        
        // Build q3 pairs inline: q3 = ql + 4*qh (unsigned 0-7)
        // Then we subtract 4 in the dequant to get signed -4 to +3
        #define BUILD_Q3_PAIR(qs, qh) \
            (((qs >> shift) & 0x3) | (((qh >> h_shift) & 1) << 2)) | \
            ((((qs >> (shift + 2)) & 0x3) | (((qh >> (h_shift + 1)) & 1) << 2)) << 16)
        
        const uint32_t q0_lo = BUILD_Q3_PAIR(qs0_lo, qh0_lo);
        const uint32_t q0_hi = BUILD_Q3_PAIR(qs0_hi, qh0_hi);
        const uint32_t q1_lo = BUILD_Q3_PAIR(qs1_lo, qh1_lo);
        const uint32_t q1_hi = BUILD_Q3_PAIR(qs1_hi, qh1_hi);
        const uint32_t q2_lo = BUILD_Q3_PAIR(qs2_lo, qh2_lo);
        const uint32_t q2_hi = BUILD_Q3_PAIR(qs2_hi, qh2_hi);
        const uint32_t q3_lo = BUILD_Q3_PAIR(qs3_lo, qh3_lo);
        const uint32_t q3_hi = BUILD_Q3_PAIR(qs3_hi, qh3_hi);
        
        #undef BUILD_Q3_PAIR
        
        // Type-specific dequant using LOP3 + hsub2 + hmul2
        // Q3_K: w = d * (q3 - 4) where q3 is unsigned 0-7
        constexpr int LO_MASK = 0x00070007;  // 3-bit mask
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64046404;  // 1024+4 = 1028.0 in fp16
            
            const half2 s0 = __half2half2(dm0), s1 = __half2half2(dm1);
            const half2 s2 = __half2half2(dm2), s3 = __half2half2(dm3);
            
            // Slice 0
            int w0 = lop3<0xEA>(q0_lo, LO_MASK, EX);
            int w1 = lop3<0xEA>(q0_hi, LO_MASK, EX);
            half2 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            half2 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 1
            w0 = lop3<0xEA>(q1_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q1_hi, LO_MASK, EX);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 2
            w0 = lop3<0xEA>(q2_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q2_hi, LO_MASK, EX);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 3
            w0 = lop3<0xEA>(q3_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q3_hi, LO_MASK, EX);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX = 0x43004300;
            constexpr uint32_t SUB = 0x43044304;  // 128+4 = 132.0 in bf16
            
            const __nv_bfloat162 s0 = __bfloat162bfloat162(__float2bfloat16(__half2float(dm0)));
            const __nv_bfloat162 s1 = __bfloat162bfloat162(__float2bfloat16(__half2float(dm1)));
            const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(__half2float(dm2)));
            const __nv_bfloat162 s3 = __bfloat162bfloat162(__float2bfloat16(__half2float(dm3)));
            
            int w0 = lop3<0xEA>(q0_lo, LO_MASK, EX);
            int w1 = lop3<0xEA>(q0_hi, LO_MASK, EX);
            __nv_bfloat162 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            __nv_bfloat162 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q1_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q1_hi, LO_MASK, EX);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q2_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q2_hi, LO_MASK, EX);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q3_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q3_hi, LO_MASK, EX);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else {
            // FP8: use FP16 path
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64046404;
            
            const half2 s0 = __half2half2(dm0), s1 = __half2half2(dm1);
            const half2 s2 = __half2half2(dm2), s3 = __half2half2(dm3);
            
            int w0 = lop3<0xEA>(q0_lo, LO_MASK, EX);
            int w1 = lop3<0xEA>(q0_hi, LO_MASK, EX);
            half2 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            half2 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q1_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q1_hi, LO_MASK, EX);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q2_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q2_hi, LO_MASK, EX);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(q3_lo, LO_MASK, EX);
            w1 = lop3<0xEA>(q3_hi, LO_MASK, EX);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
        }
    }
};

// Convenience aliases
using Q3K_Dequant_FP16 = gemx_dequant_traits<block_c_q3_K, half, half>;
using Q3K_Dequant_BF16 = gemx_dequant_traits<block_c_q3_K, __nv_bfloat16, __nv_bfloat16>;
using Q3K_Dequant_FP8 = gemx_dequant_traits<block_c_q3_K, __nv_fp8_e4m3, half>;
