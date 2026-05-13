#pragma once

// =============================================================================
// Q5_0 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 5-bit weights (4-bit low + 1-bit high)
// - qs: 32 bits = 8 × 4-bit low nibbles
// - qh: 8 bits = 8 × 1-bit high bits
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
struct vec_dot_q_loader_q5_0_inline {
    using acc2_type = acc2_for_act_t<acc_t>;
    
    int qs;                  // 8 × 4-bit low nibbles
    uint8_t qh;              // 8 × 1-bit high bits
    acc2_type dm;            // (d, m) in native format for acc_t
    
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;
    }
    
    // Compute 5-bit value: low 4 bits + high 1 bit
    __device__ __forceinline__ static int get_q5(int ql, int qh_bit) {
        return ql | (qh_bit << 4);
    }
    
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q5_0* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q5_0 uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q5_0_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q5_0_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Lookup table for qs indices in data[] array:
        // threads 0-3 → data[0-3], threads 4-7 → data[7-10],
        // threads 8-11 → data[14-17], threads 12-15 → data[20-23]
        static constexpr int qs_idx[16] = {
            0, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 20, 21, 22, 23
        };
        
        // Lookup table for qh indices in data[] array:
        // group 0 (threads 0-3) → data[4], group 1 (threads 4-7) → data[11],
        // group 2 (threads 8-11) → data[13], group 3 (threads 12-15) → data[24]
        static constexpr int qh_idx[4] = {4, 11, 13, 24};
        
        // Load qs (8 x 4-bit low nibbles)
        qs = blk->data[qs_idx[lane]];
        
        // Load qh - packed as 4 uint8_t per int, extract this thread's byte
        const int qh_group = lane >> 2;
        const int qh_in_group = lane & 3;
        const int qh_packed = blk->data[qh_idx[qh_group]];
        qh = (qh_packed >> (qh_in_group * 8)) & 0xFF;
        
        // Scale lookup - 4 scales for 16 threads (4 threads share each scale)
        // d0 → data[5] low 16 bits, d1 → data[12] low 16 bits,
        // d2 → data[12] high 16 bits, d3 → data[19] low 16 bits
        half d;
        const int scale_group = lane >> 2;
        switch (scale_group) {
            case 0: d = __ushort_as_half(blk->data[5] & 0xFFFF); break;
            case 1: d = __ushort_as_half(blk->data[12] & 0xFFFF); break;
            case 2: d = __ushort_as_half((blk->data[12] >> 16) & 0xFFFF); break;
            default: d = __ushort_as_half(blk->data[19] & 0xFFFF); break;
        }
        
        // Q5_0: dequant = d * (q - 16), so m = -16*d
        dm = convert_half2_to_acc2<acc2_type>(__halves2half2(d, __hmul(d, __float2half(-16.0f))));
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q5_0 uses 16-thread interface");
        
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH: Extract low nibbles from LOP3-ready layout, combine with high bits
            // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
            //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
            const float d = lo(dm);
            const float m = hi(dm);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // Extract low nibbles from LOP3-ready positions
            const int nl0 = q & 0xF;
            const int nl1 = (q >> 16) & 0xF;
            const int nl2 = (q >> 8) & 0xF;
            const int nl3 = (q >> 24) & 0xF;
            const int nl4 = (q >> 4) & 0xF;
            const int nl5 = (q >> 20) & 0xF;
            const int nl6 = (q >> 12) & 0xF;
            const int nl7 = (q >> 28) & 0xF;
            
            float sum;
            // Elements 0-3: combine low nibble with high bit
            {
                const float4 yv = y4[0];
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl0, (h >> 0) & 1)), m), yv.x, 0.0f);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl1, (h >> 1) & 1)), m), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl2, (h >> 2) & 1)), m), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl3, (h >> 3) & 1)), m), yv.w, sum);
            }
            // Elements 4-7
            {
                const float4 yv = y4[1];
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl4, (h >> 4) & 1)), m), yv.x, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl5, (h >> 5) & 1)), m), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl6, (h >> 6) & 1)), m), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl7, (h >> 7) & 1)), m), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + SHIFT-BASED EXTRACTION (no PRMT needed)
            // LOP3-ready layout: v → (n0,n1), v>>8 → (n2,n3), v>>4 → (n4,n5), v>>12 → (n6,n7)
            const half2 d2 = __half2half2(lo_acc2(dm));
            const half2 m2 = __half2half2(hi_acc2(dm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // Extract high bits from qh and place in bit 4 for each element pair
            // After LOP3-ready extraction:
            //   v → (n0, n1) needs h0 at bit 4 low half, h1 at bit 4 high half
            //   v >> 8 → (n2, n3) needs h2, h3
            //   v >> 4 → (n4, n5) needs h4, h5
            //   v >> 12 → (n6, n7) needs h6, h7
            const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
            const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
            const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
            const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x001f001f;  // 5-bit mask
            
            // SHIFT-BASED EXTRACTION: Direct shifts produce half2-aligned pairs
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly, add high bits
                const uint32_t q5_01 = ((uint32_t)q & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8, add high bits
                const uint32_t q5_23 = (((uint32_t)q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4, add high bits
                const uint32_t q5_45 = (((uint32_t)q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12, add high bits
                const uint32_t q5_67 = (((uint32_t)q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 + SHIFT-BASED EXTRACTION (no PRMT needed)
            // LOP3-ready layout: v → (n0,n1), v>>8 → (n2,n3), v>>4 → (n4,n5), v>>12 → (n6,n7)
            const __nv_bfloat162 d2 = __bfloat162bfloat162(lo_acc2(dm));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(hi_acc2(dm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // Extract high bits from qh and place in bit 4 for each element pair
            const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
            const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
            const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
            const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
            
            // LOP3 magic constants for BF16
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK = 0x001f001f;  // 5-bit mask
            
            // SHIFT-BASED EXTRACTION: Direct shifts produce bf162-aligned pairs
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly, add high bits
                const uint32_t q5_01 = ((uint32_t)q & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8, add high bits
                const uint32_t q5_23 = (((uint32_t)q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4, add high bits
                const uint32_t q5_45 = (((uint32_t)q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12, add high bits
                const uint32_t q5_67 = (((uint32_t)q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[3], sum2);
            }
            
            return __bfloat162float(__low2bfloat16(sum2)) + __bfloat162float(__high2bfloat16(sum2));
            
        } else {
            // =================================================================
            // FP8 SPECIALIZED PATH - High performance via FP16 accumulation
            // =================================================================
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half d_h = lo_acc2(dm);
            const half m_h = hi_acc2(dm);
            const half2 d2 = __half2half2(d_h);
            const half2 m2 = __half2half2(m_h);
            
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
            
            // SHIFT-BASED EXTRACTION with high bits (no PRMT needed)
            // LOP3-ready layout: v → (n0,n1), v>>8 → (n2,n3), v>>4 → (n4,n5), v>>12 → (n6,n7)
            const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
            const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
            const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
            const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
            
            // LOP3 weight dequantization
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x001f001f;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly, add high bits
                const uint32_t q5_01 = ((uint32_t)q & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y0, sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8, add high bits
                const uint32_t q5_23 = (((uint32_t)q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y1, sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4, add high bits
                const uint32_t q5_45 = (((uint32_t)q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2_v, sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12, add high bits
                const uint32_t q5_67 = (((uint32_t)q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y3, sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N < 16, "Q5_0 uses 16-thread interface");
        
        const float d = to_f32(lo_acc2(dm));
        const float m = to_f32(hi_acc2(dm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        // Extract low nibbles from LOP3-ready packed layout
        // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
        //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
        const int nl0 = q & 0xF;
        const int nl1 = (q >> 16) & 0xF;
        const int nl2 = (q >> 8) & 0xF;
        const int nl3 = (q >> 24) & 0xF;
        const int nl4 = (q >> 4) & 0xF;
        const int nl5 = (q >> 20) & 0xF;
        const int nl6 = (q >> 12) & 0xF;
        const int nl7 = (q >> 28) & 0xF;
        
        // Output in sequential order: out[0..7] = dequant(q5_0..q5_7)
        out4[0] = make_float4(
            __fmaf_rn(d, float(get_q5(nl0, (h >> 0) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl1, (h >> 1) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl2, (h >> 2) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl3, (h >> 3) & 1)), m)
        );
        out4[1] = make_float4(
            __fmaf_rn(d, float(get_q5(nl4, (h >> 4) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl5, (h >> 5) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl6, (h >> 6) & 1)), m),
            __fmaf_rn(d, float(get_q5(nl7, (h >> 7) & 1)), m)
        );
    }
};

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q5_0, vdr, act_t> {
    using type = vec_dot_q_loader_q5_0_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q5_0 (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q5_0, vdr, act_t> {
    using type = vec_dot_q_loader_q5_0_inline<vdr, acc_for_act_t<act_t>>;
};
// =============================================================================
// GEMX DEQUANT TRAITS - Q5_0 (5-bit symmetric: value = d * (q5 - 16))
// =============================================================================
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_0, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;  // m = -16*d (computed from d)
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q5_0>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 5;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q5_0 K/128 layout (112 bytes = 28 ints):
    //   data[0-3]=qs0-3, data[4]=qh0123, data[5]=d0|_spad, data[6]=_spad,
    //   data[7-10]=qs4-7, data[11]=qh4567, data[12]=d1|d2, data[13]=qh891011,
    //   data[14-17]=qs8-11, data[18]=_spad, data[19]=d3|_spad,
    //   data[20-23]=qs12-15, data[24]=qh12131415
    //
    // Scale groups: d0 for qs0-3, d1 for qs4-7, d2 for qs8-11, d3 for qs12-15
    // qh contains high bits: 4 threads × 8 bits = 4 bytes per qh word
    // =========================================================================
    
    static constexpr int K128_BYTES = 112;
    
    // -------------------------------------------------------------------------
    // COMPILE-TIME LANE PARAMETERS
    // -------------------------------------------------------------------------
    
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;           // Row index (0-7)
        static constexpr int K_GROUP = LANE % 4;     // Which 4-element group within K=16 (0-3)
        
        static constexpr int THREAD_IN_BLOCK = K_GROUP / 2;  // 0 or 1 for first K=16
        static constexpr int NIBBLE_HALF = K_GROUP % 2;      // 0=first 4, 1=second 4
    };
    
    // Combine low nibble with high bit to form 5-bit value
    __device__ __forceinline__ static int make_q5(int low4, int high1) {
        return low4 | (high1 << 4);
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 Q5 ELEMENTS (one FragB) with compile-time shift
    // -------------------------------------------------------------------------
    // Need to combine low nibbles from qs with high bits from qh
    // LOP3-ready layout: v >> 0 → (n0,n1), v >> 8 → (n2,n3), v >> 4 → (n4,n5), v >> 12 → (n6,n7)
    // For NIBBLE_HALF=0: extract (n0,n1,n2,n3) with h bits 0,1,2,3
    // For NIBBLE_HALF=1: extract (n4,n5,n6,n7) with h bits 4,5,6,7
    
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(int q, uint8_t h, FragB_t& frag) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int SHIFT0 = (NIBBLE_HALF == 0) ? 0 : 4;
        constexpr int SHIFT1 = (NIBBLE_HALF == 0) ? 8 : 12;
        constexpr int H_BASE = (NIBBLE_HALF == 0) ? 0 : 4;
        
        // Build high-bit pairs for half2 alignment
        // For pair (a,b): hb = (h_a << 4) | (h_b << 20)
        const uint32_t hb_01 = (((h >> (H_BASE + 0)) & 1) << 4) | (((h >> (H_BASE + 1)) & 1) << 20);
        const uint32_t hb_23 = (((h >> (H_BASE + 2)) & 1) << 4) | (((h >> (H_BASE + 3)) & 1) << 20);
        
        // Extract low nibbles and OR with high bits
        const uint32_t q5_01 = (((uint32_t)q >> SHIFT0) & 0x000f000f) | hb_01;
        const uint32_t q5_23 = (((uint32_t)q >> SHIFT1) & 0x000f000f) | hb_23;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_FP16);
            
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
    // EXTRACT 8 Q5 ELEMENTS (two FragB) for K=32 MMA (FP8)
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(int q, uint8_t h, FragB_t& frag0, FragB_t& frag1) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        // Build all high-bit pairs
        const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
        const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
        const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
        const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
        
        // Extract low nibbles and OR with high bits
        const uint32_t q5_01 = ((uint32_t)q & 0x000f000f) | hb_01;
        const uint32_t q5_23 = (((uint32_t)q >> 8) & 0x000f000f) | hb_23;
        const uint32_t q5_45 = (((uint32_t)q >> 4) & 0x000f000f) | hb_45;
        const uint32_t q5_67 = (((uint32_t)q >> 12) & 0x000f000f) | hb_67;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX);
            
            frag0[0] = __hsub2(*reinterpret_cast<half2*>(&w01), *reinterpret_cast<const half2*>(&SUB));
            frag0[1] = __hsub2(*reinterpret_cast<half2*>(&w23), *reinterpret_cast<const half2*>(&SUB));
            frag1[0] = __hsub2(*reinterpret_cast<half2*>(&w45), *reinterpret_cast<const half2*>(&SUB));
            frag1[1] = __hsub2(*reinterpret_cast<half2*>(&w67), *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_BF);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX_BF);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX_BF);
            
            frag0[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag0[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w45), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w67), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_FP16);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX_FP16);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX_FP16);
            
            half2 h01 = __hsub2(*reinterpret_cast<half2*>(&w01), *reinterpret_cast<const half2*>(&SUB_FP16));
            half2 h23 = __hsub2(*reinterpret_cast<half2*>(&w23), *reinterpret_cast<const half2*>(&SUB_FP16));
            half2 h45 = __hsub2(*reinterpret_cast<half2*>(&w45), *reinterpret_cast<const half2*>(&SUB_FP16));
            half2 h67 = __hsub2(*reinterpret_cast<half2*>(&w67), *reinterpret_cast<const half2*>(&SUB_FP16));
            
            uint16_t fp8_01 = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&h01), __NV_SATFINITE, __NV_E4M3);
            uint16_t fp8_23 = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&h23), __NV_SATFINITE, __NV_E4M3);
            uint16_t fp8_45 = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&h45), __NV_SATFINITE, __NV_E4M3);
            uint16_t fp8_67 = __nv_cvt_halfraw2_to_fp8x2(*reinterpret_cast<__half2_raw*>(&h67), __NV_SATFINITE, __NV_E4M3);
            
            frag0[0] = (static_cast<uint32_t>(fp8_23) << 16) | fp8_01;
            frag1[0] = (static_cast<uint32_t>(fp8_67) << 16) | fp8_45;
        }
    }
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // MMA m16n8k16 requires:
    //   frag[0] = half2(B[k0, n], B[k1, n]) where k0=(lane%4)*2, k1=k0+1
    //   frag[1] = half2(B[k0+8, n], B[k1+8, n])
    //
    // Q5_0 K/128 layout: 16 threads × 8 elements (5-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7: low 4 bits in qs (int), high bits in qh (uint8_t)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //
    // Q5_0 dequant: w = d * (q5 - 16) where q5 is 0-31 (symmetric)
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // ADDRESS COMPUTATION
        // =====================================================================
        const int row = lane >> 2;          // N: 0-7
        const int k_group = lane & 3;       // K_GROUP: 0-3
        
        // Base pointer for this row
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Q5_0 K/128 byte layout (112 bytes):
        // Based on struct: qs0-3, qh0123, d0, _spad0, qs4-7, qh4567, d1, d2, qh891011, qs8-11, _spad1, d3, qs12-15, qh12131415
        //
        // qs byte offsets for each thread:
        static constexpr int qs_byte_offset[16] = {
            0, 4, 8, 12,      // threads 0-3 (qs0-3)
            28, 32, 36, 40,   // threads 4-7 (qs4-7)
            56, 60, 64, 68,   // threads 8-11 (qs8-11)
            80, 84, 88, 92    // threads 12-15 (qs12-15)
        };
        // qh byte offsets: qh0123=16, qh4567=44, qh891011=52, qh12131415=96
        static constexpr int qh_byte_offset[4] = {16, 44, 52, 96};
        // d byte offsets: d0=20, d1=48, d2=50, d3=76
        static constexpr int d_byte_offset[4] = {20, 48, 50, 76};
        
        // Thread indices for this k_iter
        const int thread_lo = k_iter << 1;      // k_iter * 2
        const int thread_hi = thread_lo + 1;
        const int scale_group = thread_lo >> 2; // which scale (0-3)
        const int thread_in_group_lo = thread_lo & 3;
        const int thread_in_group_hi = thread_hi & 3;
        
        // Load both qs and qh values
        const int qs_lo = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_lo]);
        const int qs_hi = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_hi]);
        const int qh_packed = *reinterpret_cast<const int*>(row_base + qh_byte_offset[scale_group]);
        const uint8_t h_lo = (qh_packed >> (thread_in_group_lo * 8)) & 0xFF;
        const uint8_t h_hi = (qh_packed >> (thread_in_group_hi * 8)) & 0xFF;
        const half d = *reinterpret_cast<const half*>(row_base + d_byte_offset[scale_group]);
        
        // =====================================================================
        // ELEMENT EXTRACTION
        // =====================================================================
        // MMA k_group determines which K positions:
        //   k_group=0: K = 0,1,8,9
        //   k_group=1: K = 2,3,10,11
        //   k_group=2: K = 4,5,12,13
        //   k_group=3: K = 6,7,14,15
        //
        // Q5_0 nibble layout in int (same as Q4):
        //   bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //   [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        //
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // k_group → h_base: 0→0, 1→2, 2→4, 3→6
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;  // Each k_group needs 2 adjacent h bits
        
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        // Build 5-bit values: low 4 bits from qs, high bit from qh
        // For qs_lo: extract 2 nibbles (elements n, n+1 from k_group's position)
        // h_lo bits layout: bit i corresponds to element i of the 8 elements
        // For k_group 0: need bits 0,1 for elements 0,1
        // For k_group 1: need bits 2,3 for elements 2,3
        // etc.
        const uint32_t lo_nibbles = ((uint32_t)qs_lo >> shift) & 0x000f000f;
        const uint32_t hi_nibbles = ((uint32_t)qs_hi >> shift) & 0x000f000f;
        
        // High bits: bit h_base goes to element 0, bit h_base+1 goes to element 1
        const uint32_t hb_lo = (((h_lo >> h_base) & 1) << 4) | (((h_lo >> (h_base + 1)) & 1) << 20);
        const uint32_t hb_hi = (((h_hi >> h_base) & 1) << 4) | (((h_hi >> (h_base + 1)) & 1) << 20);
        
        const uint32_t q5_lo = lo_nibbles | hb_lo;
        const uint32_t q5_hi = hi_nibbles | hb_hi;
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q5_0: w = d * (q5 - 16) (symmetric quantization)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64106410;  // Subtract 1040.0 (1024+16) in fp16
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43104310;  // Subtract 144.0 (128+16) in bf16
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64106410;
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            half2 w0 = __hmul2(scale2, raw0);
            half2 w1 = __hmul2(scale2, raw1);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (OPTIMIZED)
    // =========================================================================
    // Processes 4 consecutive k_iters with vector loads and hoisted computations.
    // Key optimizations:
    // - Single shift/h_base computation (constant across all k_iters)
    // - Vector loads where alignment permits: int4 (16B), int2 (8B), int (4B)
    // - Shared qh/d loads for k_iter pairs (same scale_group)
    // - Q5_0 symmetric dequant: w = d * (q5 - 16) uses hmul2 (not hfma2)
    // =========================================================================
    
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        // =====================================================================
        // PHASE 1: LANE-CONSTANT COMPUTATIONS (hoisted from loop)
        // =====================================================================
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Shift pattern: k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;
        
        // LOP3 constants (compile-time)
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int EX_FP16 = 0x64006400;
        constexpr uint32_t SUB_FP16 = 0x64106410;  // 1024+16 for symmetric
        constexpr int EX_BF16 = 0x43004300;
        constexpr uint32_t SUB_BF16 = 0x43104310;  // 128+16 for symmetric
        
        if constexpr (half_idx == 0) {
            // =================================================================
            // HALF 0: k_iter 0,1,2,3 → threads 0-7
            // =================================================================
            // qs: threads 0-3 @ offsets 0,4,8,12 (16B aligned → int4)
            //     threads 4-7 @ offsets 28,32,36,40 (4B aligned → 4× int)
            // qh: scale_group 0 @ offset 16, scale_group 1 @ offset 44
            // d:  scale_group 0 @ offset 20, scale_group 1 @ offset 48
            
            // =====================================================================
            // PHASE 2: HOIST ALL LOADS UPFRONT (memory latency hiding)
            // =====================================================================
            const int4 qs_01 = *reinterpret_cast<const int4*>(row_base + 0);   // qs[0-3]
            const int qs_4 = *reinterpret_cast<const int*>(row_base + 28);
            const int qs_5 = *reinterpret_cast<const int*>(row_base + 32);
            const int qs_6 = *reinterpret_cast<const int*>(row_base + 36);
            const int qs_7 = *reinterpret_cast<const int*>(row_base + 40);
            
            const int qh_01 = *reinterpret_cast<const int*>(row_base + 16);
            const int qh_23 = *reinterpret_cast<const int*>(row_base + 44);
            const half d_01 = *reinterpret_cast<const half*>(row_base + 20);
            const half d_23 = *reinterpret_cast<const half*>(row_base + 48);
            
            // Extract h bytes
            const uint8_t h0 = qh_01 & 0xFF, h1 = (qh_01 >> 8) & 0xFF;
            const uint8_t h2 = (qh_01 >> 16) & 0xFF, h3 = (qh_01 >> 24) & 0xFF;
            const uint8_t h4 = qh_23 & 0xFF, h5 = (qh_23 >> 8) & 0xFF;
            const uint8_t h6 = (qh_23 >> 16) & 0xFF, h7 = (qh_23 >> 24) & 0xFF;
            
            // =====================================================================
            // PHASE 3: TYPE-SPECIFIC DEQUANTIZATION
            // =====================================================================
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 scale2_01 = __half2half2(d_01);
                const half2 scale2_23 = __half2half2(d_23);
                
                // k_iter=0: threads 0,1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1: threads 2,3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2: threads 4,5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3: threads 6,7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 scale2_01 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_01)));
                const __nv_bfloat162 scale2_23 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_23)));
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 scale2_01 = __half2half2(d_01);
                const half2 scale2_23 = __half2half2(d_23);
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else {
            // =================================================================
            // HALF 1: k_iter 4,5,6,7 → threads 8-15
            // =================================================================
            // qs: threads 8-11 @ offsets 56,60,64,68 (8B aligned → 2× int2)
            //     threads 12-15 @ offsets 80,84,88,92 (16B aligned → int4)
            // qh: scale_group 2 @ offset 52, scale_group 3 @ offset 96
            // d:  scale_group 2 @ offset 50, scale_group 3 @ offset 76
            
            const int2 qs_4a = *reinterpret_cast<const int2*>(row_base + 56);  // qs[8-9]
            const int2 qs_4b = *reinterpret_cast<const int2*>(row_base + 64);  // qs[10-11]
            const int4 qs_67 = *reinterpret_cast<const int4*>(row_base + 80);  // qs[12-15]
            
            const int qh_45 = *reinterpret_cast<const int*>(row_base + 52);
            const int qh_67 = *reinterpret_cast<const int*>(row_base + 96);
            const half d_45 = *reinterpret_cast<const half*>(row_base + 50);
            const half d_67 = *reinterpret_cast<const half*>(row_base + 76);
            
            const uint8_t h8 = qh_45 & 0xFF, h9 = (qh_45 >> 8) & 0xFF;
            const uint8_t h10 = (qh_45 >> 16) & 0xFF, h11 = (qh_45 >> 24) & 0xFF;
            const uint8_t h12 = qh_67 & 0xFF, h13 = (qh_67 >> 8) & 0xFF;
            const uint8_t h14 = (qh_67 >> 16) & 0xFF, h15 = (qh_67 >> 24) & 0xFF;
            
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 scale2_45 = __half2half2(d_45);
                const half2 scale2_67 = __half2half2(d_67);
                
                // k_iter=4: threads 8,9
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5: threads 10,11
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6: threads 12,13
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7: threads 14,15
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 scale2_45 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_45)));
                const __nv_bfloat162 scale2_67 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_67)));
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 scale2_45 = __half2half2(d_45);
                const half2 scale2_67 = __half2half2(d_67);
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
        }
    }
};

// Convenience aliases for Q5_0
using Q50_Dequant_FP16 = gemx_dequant_traits<block_c_q5_0, half, half>;
using Q50_Dequant_BF16 = gemx_dequant_traits<block_c_q5_0, __nv_bfloat16, __nv_bfloat16>;
using Q50_Dequant_FP8 = gemx_dequant_traits<block_c_q5_0, __nv_fp8_e4m3, half>;