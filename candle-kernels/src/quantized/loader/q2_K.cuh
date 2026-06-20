#pragma once

// =============================================================================
// Q2_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 2-bit weights from a uint16_t (16 bits)
//
// Q2_K formula: dequant = d * q + m  (where m is negative min)
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
struct vec_dot_q_loader_q2_K_inline {
    using acc2_type = acc2_for_act_t<acc_t>;
    
    uint16_t qs;             // 8 × 2-bit weights packed in 16 bits
    acc2_type dm;            // (d, m) in native format for acc_t
    
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;
    }
    
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q2_K* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q2_K uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q2_K_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q2_K_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Q2_K K/128 layout: [qs0,qs1], dm0, [qs2,qs3], dm1, ...
        // data[i*2] = packed qs for threads i*2 and i*2+1 (as uint16_t pair)
        // data[i*2+1] = dm for threads i*2 and i*2+1
        const int group = lane >> 1;        // which thread pair (0-7)
        const int in_group = lane & 1;      // 0 or 1 within pair
        
        // Extract uint16_t from packed int (qs0|qs1 in one int)
        const uint32_t qs_packed = static_cast<uint32_t>(blk->data[group * 2]);
        qs = (in_group == 0) ? (qs_packed & 0xFFFF) : (qs_packed >> 16);
        
        // dm is at data[group*2 + 1]
        const half2* dm_ptr = reinterpret_cast<const half2*>(&blk->data[group * 2 + 1]);
        dm = convert_half2_to_acc2<acc2_type>(*dm_ptr);
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q2_K uses 16-thread interface");
        
        const uint32_t q = qs;
        
        if constexpr (std::is_same_v<y_t, float>) {
            const float d = lo(dm);
            const float m = hi(dm);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            float sum = 0.0f;
            // Elements 0-3: crumbs from bits 0-7
            {
                const float4 yv = y4[0];
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 0) & 0x3), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 2) & 0x3), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 4) & 0x3), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 6) & 0x3), m), yv.w, sum);
            }
            // Elements 4-7: crumbs from bits 8-15
            {
                const float4 yv = y4[1];
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 8) & 0x3), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 10) & 0x3), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 12) & 0x3), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float((q >> 14) & 0x3), m), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + PRMT OPTIMIZATION
            const half2 d2 = __half2half2(lo_acc2(dm));
            const half2 m2 = __half2half2(hi_acc2(dm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // Extract crumbs (2-bit values) and place in separate bytes
            // q has 8 crumbs: bits 0-1=c0, 2-3=c1, ..., 14-15=c7
            // q2_even: byte0=c0, byte1=c2, byte2=c4, byte3=c6
            // q2_odd:  byte0=c1, byte1=c3, byte2=c5, byte3=c7
            const uint32_t q2_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t q2_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x00030003;  // 2-bit mask
            
            // STREAMING + PRMT OPTIMIZATION
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 + PRMT OPTIMIZATION
            const __nv_bfloat162 d2 = __bfloat162bfloat162(lo_acc2(dm));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(hi_acc2(dm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // Extract crumbs (same as FP16 path)
            const uint32_t q2_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t q2_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            
            // LOP3 magic constants for BF16
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK = 0x00030003;  // 2-bit mask
            
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX_BF);
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
            
            // Extract crumbs
            const uint32_t q2_even = ((q >> 0) & 0x3) |
                                     (((q >> 4) & 0x3) << 8) |
                                     (((q >> 8) & 0x3) << 16) |
                                     (((q >> 12) & 0x3) << 24);
            const uint32_t q2_odd = ((q >> 2) & 0x3) |
                                    (((q >> 6) & 0x3) << 8) |
                                    (((q >> 10) & 0x3) << 16) |
                                    (((q >> 14) & 0x3) << 24);
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x00030003;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const uint32_t pair = prmt_build_lop3_pair_0(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y0, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_1(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y1, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_2(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2_v, sum2);
            }
            {
                const uint32_t pair = prmt_build_lop3_pair_3(q2_even, q2_odd);
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(pair, LO_MASK, EX);
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
        static_assert(N < 16, "Q2_K uses 16-thread interface");
        
        const float d = to_f32(lo_acc2(dm));
        const float m = to_f32(hi_acc2(dm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = qs;
        
        out4[0] = make_float4(
            __fmaf_rn(d, float((q >> 0) & 0x3), m),
            __fmaf_rn(d, float((q >> 2) & 0x3), m),
            __fmaf_rn(d, float((q >> 4) & 0x3), m),
            __fmaf_rn(d, float((q >> 6) & 0x3), m)
        );
        out4[1] = make_float4(
            __fmaf_rn(d, float((q >> 8) & 0x3), m),
            __fmaf_rn(d, float((q >> 10) & 0x3), m),
            __fmaf_rn(d, float((q >> 12) & 0x3), m),
            __fmaf_rn(d, float((q >> 14) & 0x3), m)
        );
    }
};

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q2_K, vdr, act_t> {
    using type = vec_dot_q_loader_q2_K_inline<vdr, acc_for_act_t<act_t>>;
};

// K/128 compact format uses the same inline loader
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q2_K, vdr, act_t> {
    using type = vec_dot_q_loader_q2_K_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// SCALE EXTRACTION FOR Q2_K
// =============================================================================
//
// Q2_K K/128 format: Each K/128 block has 8 half2 scales (d,m) covering 128 elements.
// Scales are interleaved: data[g*2+1] contains the dm for threads 2g and 2g+1.
//
// =============================================================================

namespace gemx_q2_K {

// Extract all scales: row-major [N, K/128] → column-major [K/16, N]
// Each K/128 block has 8 groups × 1 scale each = 8 scales for 128 elements
// That's 1 scale per 16 elements, matching the output format
template <typename ScaleT>
__device__ inline void extract_scales_impl(
    const block_q2_K* __restrict__ x,
    ScaleT* __restrict__ scales_out,
    int nrows,
    int ncols
) {
    constexpr int ELEMENTS_PER_BLOCK = 128;
    constexpr int SCALES_PER_BLOCK = 8;  // 8 groups, 1 scale per 16 elements
    constexpr int ELEMENTS_PER_SCALE = 16;
    
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
        const int local_scale = scale_col % SCALES_PER_BLOCK;  // 0-7
        const int block_idx = row * blocks_per_row + block_col;
        
        // Reinterpret as K/128 block
        const block_c_q2_K_k128* blk = 
            reinterpret_cast<const block_c_q2_K_k128*>(&x[block_idx]);
        
        // Scale location in data[]: groups are at indices 1, 3, 5, 7, 9, 11, 13, 15
        // group g's dm is at data[g*2 + 1]
        const int group = local_scale;
        const int dm_data_idx = group * 2 + 1;
        const half2 dm = *reinterpret_cast<const half2*>(&blk->data[dm_data_idx]);
        
        // Output scale pair (d, m)
        const int dst_scale_idx = scale_col * nrows + row;
        scales_out[dst_scale_idx] = dm;
    }
}

} // namespace gemx_q2_K
// =============================================================================
// GEMX DEQUANT TRAITS - Q2_K (2-bit K-quant with scale and min)
// =============================================================================
// Include the base gemx_dequant infrastructure
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q2_K, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q2_K>::scales_per_ktile;
    static constexpr int bits_per_element = 2;
    
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // NEW MMA INTERFACE - COMPILE-TIME LANE DISPATCH
    // =========================================================================
    //
    // Q2_K K/128 block layout (64 bytes = 16 ints):
    //   [qs0(2B), qs1(2B)] dm0(4B) [qs2, qs3] dm1(4B) [qs4, qs5] dm2 [qs6, qs7] dm3
    //   [qs8, qs9] dm4 [qs10, qs11] dm5 [qs12, qs13] dm6 [qs14, qs15] dm7
    //
    // Layout via data[] array (16 ints):
    //   data[0] = qs0|qs1,  data[1] = dm0
    //   data[2] = qs2|qs3,  data[3] = dm1
    //   data[4] = qs4|qs5,  data[5] = dm2
    //   ...
    //   data[14] = qs14|qs15, data[15] = dm7
    //
    // MMA thread mapping: 32 lanes → 8 rows × 4 K-groups
    //   N = LANE / 4 (row 0-7)
    //   K_GROUP = LANE % 4 (0-3, selects which 4 elements within K=16)
    //
    // K/128 = 64 bytes per row
    // =========================================================================
    
    static constexpr int K128_BYTES = 64;

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH (affine per-16; int8_affine_per16 == true)
    // -------------------------------------------------------------------------
    // 2-bit unsigned (value = d·q + m, q in 0..3), natural bit order (element j at
    // qs bits 2j). Weights stay UNSIGNED; per-16 {d,m} applied by the split affine
    // fold (the min term uses in-kernel per-16 activation sums). Layout: group g
    // (bytes 8g..8g+7) = {qs_even@0, qs_odd@2, dm@4 (half2 {d,m})}.
    __device__ __forceinline__ static uint32_t unpack_field_int8(
        const uint8_t* __restrict__ rb, int field, int p)
    {
        const int pr = field >> 1, ip = field & 1;
        const uint32_t qs = *reinterpret_cast<const uint16_t*>(rb + 8 * pr + (ip ? 2 : 0));
        const uint32_t qss = (qs >> (2 * p)) & 0xFFu;  // 4 × 2-bit
        return ((qss & 0x03u) << 0) | ((qss & 0x0Cu) << 6)
             | ((qss & 0x30u) << 12) | ((qss & 0xC0u) << 18);
    }
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const uint8_t* rb = warp_rows + row * K128_BYTES;
        const int p = (q3 & 1) * 4;
        const int m0 = sub * 4 + (q3 >> 1);
        b_frag[0] = unpack_field_int8(rb, m0, p);
        b_frag[1] = unpack_field_int8(rb, m0 + 2, p);
    }
    // {d,m} for the lo 16-group (pair 2s, dm at byte 16s+4) and hi 16-group
    // (pair 2s+1, dm at byte 16s+12).
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        return *reinterpret_cast<const half2*>(row_block + 16 * sub + 4);
    }
    __device__ __forceinline__ static half2 sub_dm_hi(const uint8_t* __restrict__ row_block, int sub) {
        return *reinterpret_cast<const half2*>(row_block + 16 * sub + 12);
    }

    // Compile-time lane parameter extraction
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;              // Row 0-7
        static constexpr int K_GROUP = LANE % 4;        // K-group 0-3
        static constexpr int THREAD_IN_BLOCK = K_GROUP / 2;  // 0 for K_GROUP 0-1, 1 for 2-3
        static constexpr int CRUMB_HALF = K_GROUP % 2;  // Which 4 crumbs: 0=first, 1=second
    };
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS (one FragB) from uint16_t containing 8 × 2-bit crumbs
    // -------------------------------------------------------------------------
    // CRUMB_HALF=0: elements 0-3 (crumbs c0,c1,c2,c3)
    // CRUMB_HALF=1: elements 4-7 (crumbs c4,c5,c6,c7)
    //
    // Q2_K stores 8 × 2-bit crumbs in 16 bits:
    //   bits[1:0]=c0, bits[3:2]=c1, ..., bits[15:14]=c7
    //
    // OPTIMIZED: Direct shift extraction without expensive q2_even/q2_odd
    // For CRUMB_HALF=0: extract crumbs at positions 0,1,2,3 (bits 0-7)
    // For CRUMB_HALF=1: extract crumbs at positions 4,5,6,7 (bits 8-15)
    
    template <int CRUMB_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(uint16_t qs, FragB_t& frag) {
        const uint32_t q = qs;
        
        // Base shift: CRUMB_HALF=0 → start at bit 0, CRUMB_HALF=1 → start at bit 8
        constexpr int BASE_SHIFT = CRUMB_HALF * 8;
        
        // Extract 4 crumbs as two pairs: (c0,c1) and (c2,c3)
        // Each pair is packed as ((c_odd << 16) | c_even) for LOP3
        const uint32_t pair0 = ((q >> (BASE_SHIFT + 0)) & 0x3) | 
                               (((q >> (BASE_SHIFT + 2)) & 0x3) << 16);
        const uint32_t pair1 = ((q >> (BASE_SHIFT + 4)) & 0x3) | 
                               (((q >> (BASE_SHIFT + 6)) & 0x3) << 16);
        
        constexpr int LO_MASK = 0x00030003;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(pair0, LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(pair1, LO_MASK, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(pair0, LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(pair1, LO_MASK, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(pair0, LO_MASK, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(pair1, LO_MASK, EX_FP16);
            
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
    // EXTRACT 8 ELEMENTS (two FragB) for K=32 MMA (FP8)
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(uint16_t qs, FragB_t& frag0, FragB_t& frag1) {
        extract_4_elements<0>(qs, frag0);
        extract_4_elements<1>(qs, frag1);
    }
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // MMA m16n8k16 requires:
    //   frag[0] = half2(B[k0, n], B[k1, n]) where k0=(lane%4)*2, k1=k0+1
    //   frag[1] = half2(B[k0+8, n], B[k1+8, n])
    //
    // Q2_K K/128 layout: 16 threads × 8 elements (2-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7 packed in uint16_t
    // Layout: [qs0|qs1][dm0][qs2|qs3][dm1]... (pairs of threads share scale)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
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
        
        // Base pointer for this row
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        const int* data_ptr = reinterpret_cast<const int*>(row_base);
        
        // Q2_K block layout (64 bytes = 16 ints):
        // [qs0|qs1] dm0 [qs2|qs3] dm1 [qs4|qs5] dm2 [qs6|qs7] dm3
        // [qs8|qs9] dm4 [qs10|qs11] dm5 [qs12|qs13] dm6 [qs14|qs15] dm7
        //
        // data[k_iter*2] = qs_{k_iter*2} | qs_{k_iter*2+1}
        // data[k_iter*2+1] = dm_{k_iter}
        
        const int data_idx = k_iter << 1;  // k_iter * 2
        const uint32_t qs_packed = static_cast<uint32_t>(data_ptr[data_idx]);
        const uint16_t qs_lo = qs_packed & 0xFFFF;         // thread k_iter*2: elements 0-7
        const uint16_t qs_hi = (qs_packed >> 16) & 0xFFFF; // thread k_iter*2+1: elements 8-15
        const half2 dm = *reinterpret_cast<const half2*>(&data_ptr[data_idx + 1]);
        
        // =====================================================================
        // ELEMENT EXTRACTION
        // =====================================================================
        // MMA k_group determines which K positions:
        //   k_group=0: K = 0,1,8,9
        //   k_group=1: K = 2,3,10,11
        //   k_group=2: K = 4,5,12,13
        //   k_group=3: K = 6,7,14,15
        //
        // From qs_lo (elements 0-7): extract element pair at k_group*2, k_group*2+1
        // From qs_hi (elements 8-15): same indices relative to qs_hi (which holds 8-15)
        
        const int shift = k_group * 4;  // 2 bits per element, 2 elements = 4 bits shift
        const uint32_t q_lo_pair = ((qs_lo >> shift) & 0x3) | (((qs_lo >> (shift + 2)) & 0x3) << 16);
        const uint32_t q_hi_pair = ((qs_hi >> shift) & 0x3) | (((qs_hi >> (shift + 2)) & 0x3) << 16);
        
        // =====================================================================
        // LOP3+HSUB with scale/min application
        // Q2_K: w = scale * raw + neg_min
        // =====================================================================
        constexpr int LO_MASK = 0x00030003;
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            const half2 scale2 = __half2half2(__low2half(dm));
            const half2 neg_min2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hfma2(scale2, raw0, neg_min2);
            frag[1] = __hfma2(scale2, raw1, neg_min2);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm))));
            const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm))));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hfma2(scale2, raw0, neg_min2);
            frag[1] = __hfma2(scale2, raw1, neg_min2);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            const half2 scale2 = __half2half2(__low2half(dm));
            const half2 neg_min2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo_pair, LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi_pair, LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            half2 w0 = __hfma2(scale2, raw0, neg_min2);
            half2 w1 = __hfma2(scale2, raw1, neg_min2);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY OPTIMIZED)
    // =========================================================================
    //
    // OPTIMIZATIONS vs PREVIOUS VERSION:
    // 1. DIRECT CRUMB EXTRACTION - No expensive q2_even/q2_odd construction
    //    Old: 24 ops to build interleaved arrays
    //    New: 4 shifts + 4 ANDs per slice (inline, no array building)
    //
    // 2. FUSED SHIFT+MASK - Single operation extracts crumb pair
    //    Uses ((qs >> shift) & 0xF) to get 2 adjacent crumbs as nibble
    //    Then split into LOP3-ready format
    //
    // 3. PRMT FOR HALF-WORD CONSTRUCTION - Build (c0, c1) pairs in one op
    //    prmt(lo, hi, selector) places bytes from lo/hi into result
    //
    // 4. SCALE REUSE - Each dm covers all 4 slices, broadcast once
    //    Old: 4× scale broadcast per half
    //    New: Still 4× but with better register allocation
    //
    // Q2_K K/128 layout (64 bytes = 16 ints):
    //   [qs0|qs1] dm0 [qs2|qs3] dm1 [qs4|qs5] dm2 [qs6|qs7] dm3
    //   [qs8|qs9] dm4 [qs10|qs11] dm5 [qs12|qs13] dm6 [qs14|qs15] dm7
    //
    // Crumb layout in qs (16 bits = 8 × 2-bit crumbs):
    //   bits[1:0]=c0, [3:2]=c1, [5:4]=c2, [7:6]=c3
    //   bits[9:8]=c4, [11:10]=c5, [13:12]=c6, [15:14]=c7
    //
    // MMA k_group g needs elements {g*2, g*2+1} from lo and hi
    // Shift by g*4 extracts the right nibble (2 crumbs)
    // =========================================================================
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        // =====================================================================
        // PHASE 1: LANE MATH (computed ONCE for all 4 slices)
        // =====================================================================
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        const int* data_ptr = reinterpret_cast<const int*>(row_base);
        
        // Shift to extract crumb pair: k_group * 4 bits
        // (each crumb is 2 bits, we want 2 consecutive crumbs = 4 bits)
        const int shift = k_group << 2;
        
        // =====================================================================
        // PHASE 2: VECTORIZED LOADS (2× int4 = 32 bytes for 4 slices)
        // =====================================================================
        constexpr int base_data_idx = half_idx * 8;
        const int4* vec_ptr = reinterpret_cast<const int4*>(data_ptr + base_data_idx);
        const int4 d0 = vec_ptr[0];  // Slices 0,1: {qs01, dm0, qs23, dm1}
        const int4 d1 = vec_ptr[1];  // Slices 2,3: {qs45, dm2, qs67, dm3}
        
        // =====================================================================
        // PHASE 3: TYPE-SPECIFIC DEQUANTIZATION
        // =====================================================================
        // Q2_K: w = scale * q + neg_min (asymmetric with min value)
        //
        // For each slice:
        //   1. Extract crumb pair from lo half (c_{g*2}, c_{g*2+1})
        //   2. Extract crumb pair from hi half (same positions)
        //   3. Pack into LOP3-ready format: ((c1 << 16) | c0)
        //   4. LOP3 to FP16/BF16
        //   5. HSUB to remove bias
        //   6. HFMA with scale + neg_min
        //
        // Optimized crumb extraction:
        //   nib = (qs >> shift) & 0xF  -> gets 2 adjacent crumbs as 4-bit nibble
        //   c0 = nib & 0x3, c1 = (nib >> 2) & 0x3
        //   pair = c0 | (c1 << 16)
        // =====================================================================
        
        constexpr uint32_t CRUMB_MASK = 0x00030003;  // 2-bit mask in half-word positions
        
        // Inline helper: extract crumb pair from qs word (16-bit) at given shift
        // Returns ((c1 << 16) | c0) ready for LOP3
        #define EXTRACT_CRUMB_PAIR(qs16, shift) \
            (((qs16) >> (shift)) & 0x3) | ((((qs16) >> ((shift) + 2)) & 0x3) << 16)
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // --- Slice 0: qs=d0.x, dm=d0.y ---
            {
                const uint32_t qs = d0.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d0.y);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 1: qs=d0.z, dm=d0.w ---
            {
                const uint32_t qs = d0.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d0.w);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 2: qs=d1.x, dm=d1.y ---
            {
                const uint32_t qs = d1.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d1.y);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 3: qs=d1.z, dm=d1.w ---
            {
                const uint32_t qs = d1.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d1.w);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            // --- Slice 0 ---
            {
                const uint32_t qs = d0.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm_h = *reinterpret_cast<const half2*>(&d0.y);
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_h))));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_h))));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX_BF);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX_BF);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 1 ---
            {
                const uint32_t qs = d0.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm_h = *reinterpret_cast<const half2*>(&d0.w);
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_h))));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_h))));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX_BF);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX_BF);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 2 ---
            {
                const uint32_t qs = d1.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm_h = *reinterpret_cast<const half2*>(&d1.y);
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_h))));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_h))));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX_BF);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX_BF);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 3 ---
            {
                const uint32_t qs = d1.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm_h = *reinterpret_cast<const half2*>(&d1.w);
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_h))));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_h))));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX_BF);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX_BF);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // --- Slice 0 ---
            {
                const uint32_t qs = d0.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d0.y);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 1 ---
            {
                const uint32_t qs = d0.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d0.w);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 2 ---
            {
                const uint32_t qs = d1.x;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d1.y);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // --- Slice 3 ---
            {
                const uint32_t qs = d1.z;
                const uint32_t q_lo = EXTRACT_CRUMB_PAIR(qs & 0xFFFF, shift);
                const uint32_t q_hi = EXTRACT_CRUMB_PAIR(qs >> 16, shift);
                
                const half2 dm = *reinterpret_cast<const half2*>(&d1.w);
                const half2 scale2 = __half2half2(__low2half(dm));
                const half2 neg_min2 = __half2half2(__high2half(dm));
                
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_lo, CRUMB_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q_hi, CRUMB_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                half2 w0 = __hfma2(scale2, raw0, neg_min2);
                half2 w1 = __hfma2(scale2, raw1, neg_min2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
        }
        
        #undef EXTRACT_CRUMB_PAIR
    }
};

// Convenience aliases
using Q2K_Dequant_FP16 = gemx_dequant_traits<block_c_q2_K, half, half>;
using Q2K_Dequant_BF16 = gemx_dequant_traits<block_c_q2_K, __nv_bfloat16, __nv_bfloat16>;
using Q2K_Dequant_FP8 = gemx_dequant_traits<block_c_q2_K, __nv_fp8_e4m3, half>;