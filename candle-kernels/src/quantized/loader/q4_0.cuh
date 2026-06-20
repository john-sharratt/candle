#pragma once

// =============================================================================
// Q4_0 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 4-bit weights from a single int (32 bits)
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
struct vec_dot_q_loader_q4_0_inline {
    using acc2_type = acc2_for_act_t<acc_t>;
    
    int v;                   // 8 × 4-bit weights packed in 32 bits
    acc2_type dm;            // (d, m) in native format for acc_t
    
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;
    }
    
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q4_0* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q4_0 uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q4_0_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q4_0_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Q4_0 K/128 layout (80 bytes = 20 ints):
        // data[0-4]=qs0-4, data[5]=d0|d1, data[6-8]=qs5-7, data[9-10]=pad,
        // data[11-13]=qs8-10, data[14]=d2|d3, data[15-19]=qs11-15
        static constexpr int qs_idx[16] = {0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19};
        v = blk->data[qs_idx[lane]];
        
        // Scales: d0|d1 at data[5], d2|d3 at data[14]
        // threads 0-3 → d0 (low half of data[5])
        // threads 4-7 → d1 (high half of data[5])
        // threads 8-11 → d2 (low half of data[14])
        // threads 12-15 → d3 (high half of data[14])
        const int scale_group = lane >> 2;  // 0-3
        const int scale_data_idx = (scale_group < 2) ? 5 : 14;
        const int use_high_half = scale_group & 1;
        const half* d_base = reinterpret_cast<const half*>(&blk->data[scale_data_idx]);
        const half d = d_base[use_high_half];
        // Q4_0: dequant = d * (q - 8), so m = -8*d
        dm = convert_half2_to_acc2<acc2_type>(__halves2half2(d, __hmul(d, __float2half(-8.0f))));
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q4_0 uses 16-thread interface");
        
        const uint32_t q = v;
        
        // LOP3-READY: Extract nibble pairs using shift-based extraction
        // Layout: bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //         [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        // With LO_MASK = 0x000f000f:
        //   q & LO_MASK          → (n0, n1)
        //   (q >> 4) & LO_MASK   → (n4, n5)
        //   (q >> 8) & LO_MASK   → (n2, n3)
        //   (q >> 12) & LO_MASK  → (n6, n7)
        
        if constexpr (std::is_same_v<y_t, float>) {
            const float d = lo(dm);
            const float m = hi(dm);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // LOP3-READY extraction of individual nibbles
            const int n0 = (q >> 0) & 0xF;
            const int n1 = (q >> 16) & 0xF;
            const int n2 = (q >> 8) & 0xF;
            const int n3 = (q >> 24) & 0xF;
            const int n4 = (q >> 4) & 0xF;
            const int n5 = (q >> 20) & 0xF;
            const int n6 = (q >> 12) & 0xF;
            const int n7 = (q >> 28) & 0xF;
            
            float sum = 0.0f;
            {
                const float4 yv = y4[0];
                sum = __fmaf_rn(__fmaf_rn(d, float(n0), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n1), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n2), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n3), m), yv.w, sum);
            }
            {
                const float4 yv = y4[1];
                sum = __fmaf_rn(__fmaf_rn(d, float(n4), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n5), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n6), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(n7), m), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3-READY DIRECT EXTRACTION
            const half2 d2 = __half2half2(lo_acc2(dm));
            const half2 m2 = __half2half2(hi_acc2(dm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // LOP3-READY: Extract consecutive pairs using shifts
            constexpr uint32_t LO_MASK = 0x000f000f;
            const uint32_t nib_01 = q & LO_MASK;               // (n0, n1)
            const uint32_t nib_23 = (q >> 8) & LO_MASK;        // (n2, n3)
            const uint32_t nib_45 = (q >> 4) & LO_MASK;        // (n4, n5)
            const uint32_t nib_67 = (q >> 12) & LO_MASK;       // (n6, n7)
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK_4BIT = 0x000f000f;
            
            // LOP3-READY: Direct conversion without PRMT
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_01, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_23, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_45, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_67, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3-READY DIRECT EXTRACTION
            const __nv_bfloat162 d2 = __bfloat162bfloat162(lo_acc2(dm));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(hi_acc2(dm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // LOP3-READY: Extract consecutive pairs using shifts
            constexpr uint32_t LO_MASK = 0x000f000f;
            const uint32_t nib_01 = q & LO_MASK;               // (n0, n1)
            const uint32_t nib_23 = (q >> 8) & LO_MASK;        // (n2, n3)
            const uint32_t nib_45 = (q >> 4) & LO_MASK;        // (n4, n5)
            const uint32_t nib_67 = (q >> 12) & LO_MASK;       // (n6, n7)
            
            // LOP3 magic constants for BF16
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK_4BIT = 0x000f000f;
            
            // LOP3-READY: Direct conversion without PRMT
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_01, LO_MASK_4BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_23, LO_MASK_4BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_45, LO_MASK_4BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_67, LO_MASK_4BIT, EX_BF);
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
            
            // LOP3-READY: Extract consecutive pairs using shifts
            constexpr uint32_t LO_MASK = 0x000f000f;
            const uint32_t nib_01 = q & LO_MASK;               // (n0, n1)
            const uint32_t nib_23 = (q >> 8) & LO_MASK;        // (n2, n3)
            const uint32_t nib_45 = (q >> 4) & LO_MASK;        // (n4, n5)
            const uint32_t nib_67 = (q >> 12) & LO_MASK;       // (n6, n7)
            
            // LOP3 weight dequantization
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK_4BIT = 0x000f000f;
            
            // LOP3-READY: Direct conversion without PRMT
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_01, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y0, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_23, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y1, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_45, LO_MASK_4BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2_v, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(nib_67, LO_MASK_4BIT, EX);
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
        static_assert(N < 16, "Q4_0 uses 16-thread interface");
        
        const float d = to_f32(lo_acc2(dm));
        const float m = to_f32(hi_acc2(dm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = v;
        
        // LOP3-READY extraction: bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //                        [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        const float n0 = __fmaf_rn(d, float((q >> 0) & 0xF), m);
        const float n1 = __fmaf_rn(d, float((q >> 16) & 0xF), m);
        const float n2 = __fmaf_rn(d, float((q >> 8) & 0xF), m);
        const float n3 = __fmaf_rn(d, float((q >> 24) & 0xF), m);
        const float n4 = __fmaf_rn(d, float((q >> 4) & 0xF), m);
        const float n5 = __fmaf_rn(d, float((q >> 20) & 0xF), m);
        const float n6 = __fmaf_rn(d, float((q >> 12) & 0xF), m);
        const float n7 = __fmaf_rn(d, float((q >> 28) & 0xF), m);
        
        out4[0] = make_float4(n0, n1, n2, n3);
        out4[1] = make_float4(n4, n5, n6, n7);
    }
};

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q4_0, vdr, act_t> {
    using type = vec_dot_q_loader_q4_0_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q4_0 (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q4_0, vdr, act_t> {
    using type = vec_dot_q_loader_q4_0_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// GEMX DEQUANT TRAITS - Q4_0 (4-bit symmetric: value = d * (q - 8))
// =============================================================================
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q4_0, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_dequant_k64 = true;  // dequant_k64_unsigned is implemented
    static constexpr bool has_min = true;  // m = -8*d (computed from d)
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q4_0>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 4;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // MMA LANE-DISPATCHED DEQUANTIZATION - FULLY COMPILE-TIME
    // =========================================================================
    //
    // Q4_0 K/128 layout (80 bytes = 20 ints):
    //   data[0-4]=qs0-4, data[5]=d0|d1 (packed half), 
    //   data[6-8]=qs5-7, data[9-10]=pad
    //   data[11-13]=qs8-10, data[14]=d2|d3 (packed half),
    //   data[15-19]=qs11-15
    //
    // Scale groups: d0 for qs0-3, d1 for qs4-7, d2 for qs8-11, d3 for qs12-15
    // Each qs is 4 bytes = 8 nibbles (8 elements in LOP3-ready order)
    //
    // =========================================================================
    
    static constexpr int K128_BYTES = 80;
    
    // Byte offset for qs indices
    static constexpr int qs_byte_offset[16] = {
        0, 4, 8, 12, 16,    // qs0-4 at data[0-4]
        24, 28, 32,         // qs5-7 at data[6-8]
        44, 48, 52,         // qs8-10 at data[11-13]
        60, 64, 68, 72, 76  // qs11-15 at data[15-19]
    };
    
    // Scale byte offsets: d0|d1 at data[5]=bytes 20-23, d2|d3 at data[14]=bytes 56-59
    // d0 = byte 20, d1 = byte 22, d2 = byte 56, d3 = byte 58
    static constexpr int scale_byte_offset[4] = {20, 22, 56, 58};

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 4-bit nibbles → n8k32 B-fragment. Nibble byte order {n0,n2,n1,n3} (low) /
    // {n4,n6,n5,n7} (high); one mask + prmt 0x3120 reorders to natural {0,1,2,3}.
    // b_frag[0]/[1] come from two qs ints. The fold applies d·C + m·Σx with
    // m = -8·d (Q4_0 centers nibbles by -8).
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_OFF[16] = {0, 4, 8, 12, 16, 24, 28, 32, 44, 48, 52, 60, 64, 68, 72, 76};
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const uint8_t* rb = warp_rows + row * K128_BYTES;
        const int sh = (q3 & 1) * 4;
        const int v0 = *reinterpret_cast<const int*>(rb + QS_OFF[sub * 4 + (q3 >> 1)]);
        const int v1 = *reinterpret_cast<const int*>(rb + QS_OFF[sub * 4 + 2 + (q3 >> 1)]);
        b_frag[0] = __byte_perm((v0 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        b_frag[1] = __byte_perm((v1 >> sh) & 0x0F0F0F0F, 0, 0x3120);
    }
    // Per-sub {scale d (low), neg_min = -8·d (high)}.
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int SCALE_OFF[4] = {20, 22, 56, 58};
        const half d = *reinterpret_cast<const half*>(row_block + SCALE_OFF[sub]);
        return __halves2half2(d, __hmul(d, __float2half(-8.0f)));
    }

    // -------------------------------------------------------------------------
    // COMPILE-TIME LANE PARAMETERS
    // -------------------------------------------------------------------------
    
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;           // Row index (0-7)
        static constexpr int K_GROUP = LANE % 4;     // Which 4-element group within K=16 (0-3)
        
        // K_GROUP 0: elements 0-3   → first qs of K/16, first half
        // K_GROUP 1: elements 4-7   → first qs of K/16, second half
        // K_GROUP 2: elements 8-11  → second qs of K/16, first half
        // K_GROUP 3: elements 12-15 → second qs of K/16, second half
        
        static constexpr int THREAD_IN_BLOCK = K_GROUP / 2;  // 0 or 1 for first K=16
        static constexpr int NIBBLE_HALF = K_GROUP % 2;      // 0=first 4, 1=second 4
    };
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS (one FragB) with compile-time shift
    // -------------------------------------------------------------------------
    // LOP3-ready layout: v >> 0 → (n0,n1), v >> 8 → (n2,n3), v >> 4 → (n4,n5), v >> 12 → (n6,n7)
    // For NIBBLE_HALF=0: extract (n0,n1,n2,n3) = shifts 0, 8
    // For NIBBLE_HALF=1: extract (n4,n5,n6,n7) = shifts 4, 12
    
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(int q, FragB_t& frag) {
        constexpr int LO_MASK = 0x000f000f;
        constexpr int SHIFT0 = (NIBBLE_HALF == 0) ? 0 : 4;
        constexpr int SHIFT1 = (NIBBLE_HALF == 0) ? 8 : 12;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX_FP16);
            
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
    __device__ __forceinline__ static void extract_8_elements(int q, FragB_t& frag0, FragB_t& frag1) {
        constexpr int LO_MASK = 0x000f000f;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX);
            
            frag0[0] = __hsub2(*reinterpret_cast<half2*>(&w01), *reinterpret_cast<const half2*>(&SUB));
            frag0[1] = __hsub2(*reinterpret_cast<half2*>(&w23), *reinterpret_cast<const half2*>(&SUB));
            frag1[0] = __hsub2(*reinterpret_cast<half2*>(&w45), *reinterpret_cast<const half2*>(&SUB));
            frag1[1] = __hsub2(*reinterpret_cast<half2*>(&w67), *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX_BF);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX_BF);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX_BF);
            
            frag0[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag0[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w45), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w67), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX_FP16);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX_FP16);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX_FP16);
            
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
    // Q4_0 K/128 layout: 16 threads × 8 elements (4-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7 packed in one int (32 bits)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //   d = shared scale for this K/16 slice (4 threads share 1 scale)
    //
    // Q4_0 dequant: w = d * (q - 8) where q is 0-15
    
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
        
        // Q4_0 K/128 byte layout (80 bytes):
        // Byte 0-19:  qs0-4 (5 ints, 20 bytes)
        // Byte 20-23: d0, d1 (2 halfs, 4 bytes)
        // Byte 24-35: qs5-7 (3 ints, 12 bytes)
        // Byte 36-43: padding (8 bytes)
        // Byte 44-55: qs8-10 (3 ints, 12 bytes)
        // Byte 56-59: d2, d3 (2 halfs, 4 bytes)
        // Byte 60-79: qs11-15 (5 ints, 20 bytes)
        //
        // qs byte offsets for each thread:
        static constexpr int qs_byte_offset[16] = {
            0, 4, 8, 12, 16,      // threads 0-4
            24, 28, 32,           // threads 5-7
            44, 48, 52,           // threads 8-10
            60, 64, 68, 72, 76    // threads 11-15
        };
        // d byte offsets: d0=20, d1=22, d2=56, d3=58
        // scale_group 0 (threads 0-3) → d0, group 1 (4-7) → d1, etc.
        static constexpr int d_byte_offset[4] = {20, 22, 56, 58};
        
        // Thread indices for this k_iter
        const int thread_lo = k_iter << 1;      // k_iter * 2
        const int thread_hi = thread_lo + 1;
        const int scale_group = thread_lo >> 2; // which scale (0-3)
        
        // Load both qs values
        const int qs_lo = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_lo]);
        const int qs_hi = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_hi]);
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
        // Q4_0 nibble layout in int:
        //   bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //   [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        //
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        constexpr int LO_MASK = 0x000f000f;
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q4_0: w = d * (q - 8) (symmetric quantization)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64086408;  // Subtract 1032.0 (1024+8) in fp16
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43084308;  // Subtract 136.0 (128+8) in bf16
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64086408;
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
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
    //
    // Q4_0 K/128 byte layout (80 bytes):
    //   Byte 0-19:  qs0-4 (5 ints, 20 bytes)
    //   Byte 20-23: d0, d1 (2 halfs, 4 bytes)
    //   Byte 24-35: qs5-7 (3 ints, 12 bytes)
    //   Byte 36-43: padding (8 bytes)
    //   Byte 44-55: qs8-10 (3 ints, 12 bytes)
    //   Byte 56-59: d2, d3 (2 halfs, 4 bytes)
    //   Byte 60-79: qs11-15 (5 ints, 20 bytes)
    //
    // half_idx=0 (k_iter 0-3, threads 0-7):
    //   k_iter=0: qs @ 0,4   → d @ 20   (int4 @ 0)
    //   k_iter=1: qs @ 8,12  → d @ 20   (int4 @ 0)
    //   k_iter=2: qs @ 16,24 → d @ 22   (2× int @ 16,24)
    //   k_iter=3: qs @ 28,32 → d @ 22   (2× int @ 28,32)
    //
    // half_idx=1 (k_iter 4-7, threads 8-15):
    //   k_iter=4: qs @ 44,48 → d @ 56   (2× int @ 44,48)
    //   k_iter=5: qs @ 52,60 → d @ 56   (2× int @ 52,60)
    //   k_iter=6: qs @ 64,68 → d @ 58   (int4 @ 64)
    //   k_iter=7: qs @ 72,76 → d @ 58   (int4 @ 64)
    //
    // Q4_0 dequant: w = d * (q - 8) (symmetric, uses __hmul2)
    // =========================================================================
    template <int half_idx>  // 0 = k16 slices 0-3, 1 = k16 slices 4-7
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        // =====================================================================
        // PHASE 1: LANE MATH (computed ONCE)
        // =====================================================================
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Precompute shift: k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // =====================================================================
        // PHASE 2: HOIST ALL LOADS UPFRONT (memory latency hiding)
        // =====================================================================
        int qs0_lo, qs0_hi, qs1_lo, qs1_hi, qs2_lo, qs2_hi, qs3_lo, qs3_hi;
        half d0, d1;
        
        if constexpr (half_idx == 0) {
            // Load scales: d0 @ 20, d1 @ 22 (load as int and extract)
            const int dm_packed = *reinterpret_cast<const int*>(row_base + 20);
            d0 = *reinterpret_cast<const half*>(&dm_packed);
            d1 = *(reinterpret_cast<const half*>(&dm_packed) + 1);
            
            // k_iter 0,1: qs @ {0,4,8,12} - int4 @ 0 (16B aligned)
            {
                const int4 qs_vec = *reinterpret_cast<const int4*>(row_base + 0);
                qs0_lo = qs_vec.x;  // offset 0
                qs0_hi = qs_vec.y;  // offset 4
                qs1_lo = qs_vec.z;  // offset 8
                qs1_hi = qs_vec.w;  // offset 12
            }
            
            // k_iter 2,3: qs @ {16,24,28,32} - 4× int (non-contiguous)
            qs2_lo = *reinterpret_cast<const int*>(row_base + 16);
            qs2_hi = *reinterpret_cast<const int*>(row_base + 24);
            qs3_lo = *reinterpret_cast<const int*>(row_base + 28);
            qs3_hi = *reinterpret_cast<const int*>(row_base + 32);
            
        } else {
            // Load scales: d2 @ 56, d3 @ 58 (load as int and extract)
            const int dm_packed = *reinterpret_cast<const int*>(row_base + 56);
            d0 = *reinterpret_cast<const half*>(&dm_packed);
            d1 = *(reinterpret_cast<const half*>(&dm_packed) + 1);
            
            // k_iter 4,5: qs @ {44,48,52,60} - 4× int (non-contiguous)
            qs0_lo = *reinterpret_cast<const int*>(row_base + 44);
            qs0_hi = *reinterpret_cast<const int*>(row_base + 48);
            qs1_lo = *reinterpret_cast<const int*>(row_base + 52);
            qs1_hi = *reinterpret_cast<const int*>(row_base + 60);
            
            // k_iter 6,7: qs @ {64,68,72,76} - int4 @ 64 (16B aligned)
            {
                const int4 qs_vec = *reinterpret_cast<const int4*>(row_base + 64);
                qs2_lo = qs_vec.x;  // offset 64
                qs2_hi = qs_vec.y;  // offset 68
                qs3_lo = qs_vec.z;  // offset 72
                qs3_hi = qs_vec.w;  // offset 76
            }
        }
        
        // =====================================================================
        // PHASE 3: TYPE-SPECIFIC DEQUANTIZATION
        // Q4_0: w = d * (q - 8) (symmetric, uses __hmul2)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64086408;  // 1024 + 8 for symmetric centering
            
            // --- SCALE GROUP 0: d0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const half2 scale2 = __half2half2(d0);
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: d1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const half2 scale2 = __half2half2(d1);
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t SUB = 0x43084308;  // 128 + 8 for symmetric centering
            
            // Convert scales once
            const float f_d0 = __half2float(d0);
            const float f_d1 = __half2float(d1);
            
            // --- SCALE GROUP 0: d0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(f_d0));
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hmul2(scale2, raw0);
                    __nv_bfloat162 w1 = __hmul2(scale2, raw1);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hmul2(scale2, raw0);
                    __nv_bfloat162 w1 = __hmul2(scale2, raw1);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: d1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(f_d1));
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hmul2(scale2, raw0);
                    __nv_bfloat162 w1 = __hmul2(scale2, raw1);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hmul2(scale2, raw0);
                    __nv_bfloat162 w1 = __hmul2(scale2, raw1);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64086408;
            
            // --- SCALE GROUP 0: d0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const half2 scale2 = __half2half2(d0);
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: d1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const half2 scale2 = __half2half2(d1);
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hmul2(scale2, raw0);
                    half2 w1 = __hmul2(scale2, raw1);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
        }
    }
    
    // =========================================================================
    // DEQUANT_K64 FOR GEMX KERNEL (K/128 → 8 FragB)
    // =========================================================================
    // This function dequantizes a full K/128 block into 8 MMA-ready fragments.
    // Used by the GEMX tensor core kernel for embedded-scale blocks.
    //
    // Output organization (for m16n8k16 MMA):
    //   frag0-frag3: K=0-63 (first K/64 half)
    //   frag4-frag7: K=64-127 (second K/64 half)
    //
    // Each FragB holds 4 elements for the current thread's MMA position.
    // Scale application is done during dequant since Q4_0 has embedded scales.
    //
    // Template parameter FragB_t allows compatibility with both GemxVec and Vec
    // fragment types used in different parts of the codebase.
    // =========================================================================
    
    template <typename FragB_t>
    __device__ __forceinline__ static void dequant_k64_unsigned(
        const block_c_q4_0& kblock,
        FragB_t& frag0, FragB_t& frag1, FragB_t& frag2, FragB_t& frag3,
        FragB_t& frag4, FragB_t& frag5, FragB_t& frag6, FragB_t& frag7
    ) {
        // Thread lane within warp determines MMA position
        const int lane = threadIdx.x % 32;
        const int n_col = lane / 4;      // N column (0-7)
        const int k_group = lane % 4;    // K position within K=16 tile (0-3)
        
        // For Q4_0 K/128: 16 qs ints, each containing 8 nibbles (8 elements)
        // qs layout: [qs0-4][d0,d1][qs5-7][pad][qs8-10][d2,d3][qs11-15]
        // Each qs covers 8 elements: qs_i → elements i*8 to i*8+7
        
        // Byte offsets in block for each qs index
        static constexpr int qs_byte[16] = {
            0, 4, 8, 12, 16,    // qs0-4 (K=0-39)
            24, 28, 32,         // qs5-7 (K=40-63)
            44, 48, 52,         // qs8-10 (K=64-87)
            60, 64, 68, 72, 76  // qs11-15 (K=88-127)
        };
        
        // Scale byte offsets: d0=byte20, d1=byte22, d2=byte56, d3=byte58
        static constexpr int scale_byte[4] = {20, 22, 56, 58};
        
        const uint8_t* block_bytes = reinterpret_cast<const uint8_t*>(&kblock);
        
        // Helper lambda to dequantize one FragB for a given K-tile
        // k_tile: 0-7 (which K=16 tile within K/128)
        // Each thread extracts 4 elements based on k_group
        auto dequant_frag = [&](int k_tile, FragB_t& frag) {
            
            // Which qs index for this k_tile and k_group?
            // k_tile 0: K=0-15  → qs0 (k_group 0,1) or qs1 (k_group 2,3)
            // k_tile 1: K=16-31 → qs2 (k_group 0,1) or qs3 (k_group 2,3)
            // etc.
            int qs_idx = k_tile * 2 + (k_group / 2);
            
            // Load the packed int
            const int q = *reinterpret_cast<const int*>(block_bytes + qs_byte[qs_idx]);
            
            // Get scale for this K range
            // Scale groups: d0 for K=0-31, d1 for K=32-63, d2 for K=64-95, d3 for K=96-127
            int scale_group = k_tile / 2;
            const half d = *reinterpret_cast<const half*>(block_bytes + scale_byte[scale_group]);
            
            // Nibble half: k_group % 2 determines which 4 nibbles
            // Use compile-time constants via ternary - NVCC optimizes well
            int nibble_half = k_group % 2;
            const int SHIFT0 = (nibble_half == 0) ? 0 : 4;
            const int SHIFT1 = (nibble_half == 0) ? 8 : 12;
            
            // Dequant: extract 4 nibbles, convert to compute_t, apply scale
            constexpr int LO_MASK = 0x000f000f;
            
            if constexpr (std::is_same_v<compute_t, half>) {
                constexpr int EX = 0x64006400;
                constexpr uint32_t BIAS = 0x64086408;  // 1024 + 8 for centering at -8
                
                int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX);
                int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX);
                
                // Subtract bias to center at -8 (Q4_0: value = d * (q - 8))
                half2 h01 = __hsub2(*reinterpret_cast<half2*>(&w01),
                                    *reinterpret_cast<const half2*>(&BIAS));
                half2 h23 = __hsub2(*reinterpret_cast<half2*>(&w23),
                                    *reinterpret_cast<const half2*>(&BIAS));
                
                // Apply scale
                half2 scale2 = __half2half2(d);
                frag[0] = __hmul2(h01, scale2);
                frag[1] = __hmul2(h23, scale2);
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                constexpr int EX_BF = 0x43004300;
                constexpr uint32_t BIAS_BF = 0x43084308;  // BF16 128 + 8
                
                int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX_BF);
                int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX_BF);
                
                __nv_bfloat162 h01 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                                             *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                __nv_bfloat162 h23 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                                             *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                
                __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d)));
                frag[0] = __hmul2(h01, scale2);
                frag[1] = __hmul2(h23, scale2);
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                // FP8: dequant via FP16 then convert
                constexpr int EX = 0x64006400;
                constexpr uint32_t BIAS = 0x64086408;
                
                int w01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT0), LO_MASK, EX);
                int w23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> SHIFT1), LO_MASK, EX);
                
                half2 h01 = __hsub2(*reinterpret_cast<half2*>(&w01),
                                    *reinterpret_cast<const half2*>(&BIAS));
                half2 h23 = __hsub2(*reinterpret_cast<half2*>(&w23),
                                    *reinterpret_cast<const half2*>(&BIAS));
                
                half2 scale2 = __half2half2(d);
                h01 = __hmul2(h01, scale2);
                h23 = __hmul2(h23, scale2);
                
                // Convert to FP8
                uint16_t fp8_01 = __nv_cvt_halfraw2_to_fp8x2(
                    *reinterpret_cast<__half2_raw*>(&h01), __NV_SATFINITE, __NV_E4M3);
                uint16_t fp8_23 = __nv_cvt_halfraw2_to_fp8x2(
                    *reinterpret_cast<__half2_raw*>(&h23), __NV_SATFINITE, __NV_E4M3);
                
                frag[0] = (static_cast<uint32_t>(fp8_23) << 16) | fp8_01;
            }
        };
        
        // Dequant all 8 K-tiles
        dequant_frag(0, frag0);  // K=0-15
        dequant_frag(1, frag1);  // K=16-31
        dequant_frag(2, frag2);  // K=32-47
        dequant_frag(3, frag3);  // K=48-63
        dequant_frag(4, frag4);  // K=64-79
        dequant_frag(5, frag5);  // K=80-95
        dequant_frag(6, frag6);  // K=96-111
        dequant_frag(7, frag7);  // K=112-127
    }
};

// Convenience aliases for Q4_0
using Q40_Dequant_FP16 = gemx_dequant_traits<block_c_q4_0, half, half>;
using Q40_Dequant_BF16 = gemx_dequant_traits<block_c_q4_0, __nv_bfloat16, __nv_bfloat16>;
using Q40_Dequant_FP8 = gemx_dequant_traits<block_c_q4_0, __nv_fp8_e4m3, half>;
