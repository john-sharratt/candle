#pragma once

// =============================================================================
// Q8_0 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// This loader processes Q8_0 weights in K/128 format [K/128, N].
// Each block contains 128 × 8-bit signed quants plus embedded half2 scales.
//
// Q8_0 is 8-bit SYMMETRIC quantization: value = d * q8
// where q8 ∈ [-128, 127] (int8_t) and d is the scale.
//
// LAYOUT
// ------
// Weights: [K/128, N] block_c_q8_0_k128
//   - qs0..qs3 (int4): 16 × int8_t per 16-element slice
// Scales:  embedded half2 (d0, d1) for each 32-element half-block
//
// OPTIMIZATIONS (matching Q4_K reference)
// ---------------------------------------
// 1. NATIVE ACC STORAGE: scale stored in native acc type
//    - float for F32, half for F16, bfloat16 for BF16
//    - Avoids runtime conversion overhead
//
// 2. CONSTEXPR IF DISPATCH: type-specialized paths via single template
//    - F32: float4 vector loads, dp4a-style accumulation
//    - F16: half2 vector loads, __hfma2 for 2× throughput
//    - BF16: bfloat162 vector loads, __hfma2 for 2× throughput
//    - FP8/other: scalar fallback
//
// 3. Q8_0 SIMPLICITY: Direct int8→float conversion (no nibble extraction)
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
struct vec_dot_q_loader_q8_0_inline {
    // -------------------------------------------------------------------------
    // STRUCT FIELDS - K/128: 8 elements per thread = int2
    // -------------------------------------------------------------------------
    int2 qs;             // 8B - 8 × int8_t quants
    acc_t d_x;           // Scale in native format for acc_t
    
    // -------------------------------------------------------------------------
    // THREAD INDEX HELPERS
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;  // 0-15 for K/128
    }
    
    // -------------------------------------------------------------------------
    // 4-PART LOAD INTERFACE (16 elements per part)
    // -------------------------------------------------------------------------
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q8_0* __restrict__ x,
        const int row,
        const int kbx,
        const int num_rows
    ) {
        static_assert(N == 0, "Q8_0 uses 1-part interface for K/128");

        const int block_idx = kbx * num_rows + row;
        const block_c_q8_0_k128* __restrict__ blk = reinterpret_cast<const block_c_q8_0_k128*>(&x[block_idx]);

        // Q8_0 K/128 layout (36 ints total):
        // data[0-1]=qs0, data[2-3]=qs1, data[4-5]=qs2, data[6]=d0|d1, data[7]=pad
        // data[8-9]=qs3, data[10-11]=qs4, data[12-13]=qs5, data[14-15]=qs6
        // data[16-17]=qs7, data[18-19]=qs8, data[20-21]=qs9, data[22-23]=qs10
        // data[24-25]=qs11, data[26-27]=qs12, data[28]=d2|d3, data[29]=pad
        // data[30-31]=qs13, data[32-33]=qs14, data[34-35]=qs15
        constexpr int qs_idx[16] = {0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34};
        constexpr int d_idx[16]  = {6, 6, 6, 6, 6, 6, 6, 6, 28, 28, 28, 28, 28, 28, 28, 28};
        // Threads 0-3 use d0 (low half of data[6]), 4-7 use d1 (high half of data[6])
        // Threads 8-11 use d2 (low half of data[28]), 12-15 use d3 (high half of data[28])
        constexpr int d_shift[16] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};

        const int lane = get_lane();
        
        // Load 8 int8_t quants as int2
        qs.x = blk->data[qs_idx[lane]];
        qs.y = blk->data[qs_idx[lane] + 1];

        // Load scale (half from packed half2)
        const half2 d2 = *reinterpret_cast<const half2*>(&blk->data[d_idx[lane]]);
        const half d_half = (d_shift[lane] == 0) ? __low2half(d2) : __high2half(d2);
        d_x = to_acc<acc_t>(d_half);
    }
    
    // -------------------------------------------------------------------------
    // DOT PRODUCT - Single template, type-specialized internally (K/128: 8 elements)
    // -------------------------------------------------------------------------
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N == 0, "Q8_0 uses 1-part interface for K/128");
        
        // Cast int2 to int8_t array for element access (8 elements)
        const int8_t* q8 = reinterpret_cast<const int8_t*>(&qs);
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH: float4 vector loads, 8 elements
            const float d = to_f32(d_x);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            float sum = 0.0f;
            
            // Elements 0-3
            {
                const float4 yv = y4[0];
                sum += float(q8[0]) * yv.x;
                sum += float(q8[1]) * yv.y;
                sum += float(q8[2]) * yv.z;
                sum += float(q8[3]) * yv.w;
            }
            // Elements 4-7
            {
                const float4 yv = y4[1];
                sum += float(q8[4]) * yv.x;
                sum += float(q8[5]) * yv.y;
                sum += float(q8[6]) * yv.z;
                sum += float(q8[7]) * yv.w;
            }
            return d * sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH: half2 vector loads, __hfma2 for 2× throughput
            const half2 d2 = __half2half2(d_x);
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            half2 sum2 = __float2half2_rn(0.0f);
            
            // Elements 0-1
            {
                const half2 yv = y2[0];
                const half2 q = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 2-3
            {
                const half2 yv = y2[1];
                const half2 q = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 4-5
            {
                const half2 yv = y2[2];
                const half2 q = __halves2half2(__int2half_rn(q8[4]), __int2half_rn(q8[5]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 6-7
            {
                const half2 yv = y2[3];
                const half2 q = __halves2half2(__int2half_rn(q8[6]), __int2half_rn(q8[7]));
                sum2 = __hfma2(q, yv, sum2);
            }
            
            // Scale and reduce
            half2 result2 = __hmul2(d2, sum2);
            return __half2float(__low2half(result2)) + __half2float(__high2half(result2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH: bfloat162 vector loads, __hfma2 for 2× throughput
            const __nv_bfloat162 d2 = __bfloat162bfloat162(d_x);
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            
            // Elements 0-1
            {
                const __nv_bfloat162 yv = y2[0];
                const __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn(q8[0]), __int2bfloat16_rn(q8[1]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 2-3
            {
                const __nv_bfloat162 yv = y2[1];
                const __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn(q8[2]), __int2bfloat16_rn(q8[3]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 4-5
            {
                const __nv_bfloat162 yv = y2[2];
                const __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn(q8[4]), __int2bfloat16_rn(q8[5]));
                sum2 = __hfma2(q, yv, sum2);
            }
            // Elements 6-7
            {
                const __nv_bfloat162 yv = y2[3];
                const __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn(q8[6]), __int2bfloat16_rn(q8[7]));
                sum2 = __hfma2(q, yv, sum2);
            }
            
            // Scale and reduce
            __nv_bfloat162 result2 = __hmul2(d2, sum2);
            return __bfloat162float(__low2bfloat16(result2)) + __bfloat162float(__high2bfloat16(result2));
            
        } else {
            // =================================================================
            // FP8 SPECIALIZED PATH - High performance via FP16 accumulation
            // =================================================================
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half2 d2 = __half2half2(d_x);
            
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
            
            // For Q8_0, we keep __int2half_rn since int8→half is efficient
            // (LOP3 optimization is more beneficial for packed nibbles)
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const half2 w = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
                sum2 = __hfma2(w, y0, sum2);
            }
            {
                const half2 w = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
                sum2 = __hfma2(w, y1, sum2);
            }
            {
                const half2 w = __halves2half2(__int2half_rn(q8[4]), __int2half_rn(q8[5]));
                sum2 = __hfma2(w, y2_v, sum2);
            }
            {
                const half2 w = __halves2half2(__int2half_rn(q8[6]), __int2half_rn(q8[7]));
                sum2 = __hfma2(w, y3, sum2);
            }
            
            // Scale and reduce
            half2 result2 = __hmul2(d2, sum2);
            return __half2float(__low2half(result2)) + __half2float(__high2half(result2));
        }
    }
    
    // -------------------------------------------------------------------------
    // DEQUANTIZE - K-TILE-MAJOR OUTPUT (float4 for efficiency, K/128: 8 elements)
    // -------------------------------------------------------------------------
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N == 0, "Q8_0 uses 1-part interface for K/128");
        
        const int y_base = get_lane() * 8;
        const float d = to_f32(d_x);
        
        const int8_t* q8 = reinterpret_cast<const int8_t*>(&qs);
        float4* out4 = reinterpret_cast<float4*>(out + y_base);
        
        // Elements 0-3
        out4[0] = make_float4(d * float(q8[0]), d * float(q8[1]), 
                               d * float(q8[2]), d * float(q8[3]));
        // Elements 4-7
        out4[1] = make_float4(d * float(q8[4]), d * float(q8[5]),
                               d * float(q8[6]), d * float(q8[7]));
    }
};

// =============================================================================
// TRAIT SPECIALIZATION
// =============================================================================
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q8_0, vdr, act_t> {
    using type = vec_dot_q_loader_q8_0_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q8_0 (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q8_0, vdr, act_t> {
    using type = vec_dot_q_loader_q8_0_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// GEMX DEQUANT TRAITS - Q8_0 (8-bit symmetric: value = d * q8)
// =============================================================================
#include "gemx_dequant.cuh"

// Forward declaration for the optimized 4x standalone function (defined below)
template <typename compute_t, int half_idx>
__device__ __forceinline__ void dequant_q8_0_for_4x_mma_k16_runtime(
    const uint8_t* __restrict__ smem_rows, int lane, uint32_t* frag_b);

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q8_0, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = false;  // Symmetric quantization
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q8_0>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 8;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q8_0 K/128 layout (144 bytes = 36 ints):
    //   Each thread has 8 int8_t quants per K-tile
    //   Scales are half2 (d0|d1, d2|d3) embedded in the data
    //
    // Layout (36 ints):
    //   data[0-1]=qs0, data[2-3]=qs1, data[4-5]=qs2, data[6]=d0|d1, data[7]=pad
    //   data[8-9]=qs3, data[10-11]=qs4, data[12-13]=qs5, data[14-15]=qs6
    //   data[16-17]=qs7, data[18-19]=qs8, data[20-21]=qs9, data[22-23]=qs10
    //   data[24-25]=qs11, data[26-27]=qs12, data[28]=d2|d3, data[29]=pad
    //   data[30-31]=qs13, data[32-33]=qs14, data[34-35]=qs15
    //
    // Scale groups: d0 for qs0-2, d1 for qs3-6, d2 for qs7-10+qs11-12, d3 for qs13-15
    // =========================================================================
    
    static constexpr int K128_BYTES = 144;

    // -------------------------------------------------------------------------
    // INT8 PATH (grouped_tc_int8): raw signed int8 quants → n8k32 B-fragment. Q8_0
    // stores q8 as signed int8 (value = d·q8), so there is NO nibble extraction or
    // centering — the 4 contiguous int8 go straight into the fragment and the fold
    // applies d·C (neg_min = 0). qs are natural-K-order within each 8-element field;
    // QS_BYTE accounts for the d|d + pad gaps in the K/128 layout.
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_BYTE[16] =
            {0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 120, 128, 136};
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const uint8_t* rb = warp_rows + row * K128_BYTES;
        const int byte_off = (q3 & 1) * 4;  // which int of the 8-elem qs field
        b_frag[0] = *reinterpret_cast<const uint32_t*>(rb + QS_BYTE[sub * 4 + (q3 >> 1)] + byte_off);
        b_frag[1] = *reinterpret_cast<const uint32_t*>(rb + QS_BYTE[sub * 4 + 2 + (q3 >> 1)] + byte_off);
    }

    // Per-sub {scale d (low), neg_min (high) = 0} — Q8_0 is symmetric.
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int SCALE_OFF[4] = {24, 26, 112, 114};  // byte offset of d0..d3
        const half d = *reinterpret_cast<const half*>(row_block + SCALE_OFF[sub]);
        return __halves2half2(d, __float2half(0.f));
    }

    // =========================================================================
    // SHARED LOOKUP TABLES - Used by both k16 and 2x_k16 dequant functions
    // =========================================================================
    // qs_offset[t] = t*8 + 8*(t>=3) + 8*(t>=13) for t=0..15
    // Packed as uint8 pairs: [qs_lo, qs_hi] for each k_iter
    // k_iter 0: t_lo=0,t_hi=1 → [0,8]    k_iter 4: t_lo=8,t_hi=9   → [72,80]
    // k_iter 1: t_lo=2,t_hi=3 → [16,32]  k_iter 5: t_lo=10,t_hi=11 → [88,96]
    // k_iter 2: t_lo=4,t_hi=5 → [40,48]  k_iter 6: t_lo=12,t_hi=13 → [104,120]
    // k_iter 3: t_lo=6,t_hi=7 → [56,64]  k_iter 7: t_lo=14,t_hi=15 → [128,136]
    // scale_offset: k_iter>>1 indexes into {24,26,112,114}
    //
    // NOTE: These are defined as local constexpr inside each function because
    // CUDA device code cannot resolve static constexpr class members.
    // LUT values: QS_LO[8] = {0, 16, 40, 56, 72, 88, 104, 128}
    //             QS_HI[8] = {8, 32, 48, 64, 80, 96, 120, 136}
    //             SCALE_OFF[4] = {24, 26, 112, 114}
    
    // -------------------------------------------------------------------------
    // COMPILE-TIME LANE PARAMETERS
    // -------------------------------------------------------------------------
    
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;           // Row index (0-7)
        static constexpr int K_GROUP = LANE % 4;     // Which 4-element group within K=16 (0-3)
        
        static constexpr int ELEMENT_BASE = K_GROUP;  // Starting element index (0-3)
    };
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 Q8 ELEMENTS (one FragB) - much simpler than 4/5-bit
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(const int8_t* q8, FragB_t& frag) {
        if constexpr (std::is_same_v<compute_t, half>) {
            // Convert 4 int8_t to half2
            frag[0] = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
            frag[1] = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            // Convert 4 int8_t to bfloat162
            frag[0] = __halves2bfloat162(__int2bfloat16_rn(q8[0]), __int2bfloat16_rn(q8[1]));
            frag[1] = __halves2bfloat162(__int2bfloat16_rn(q8[2]), __int2bfloat16_rn(q8[3]));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            // FP8 compute type uses FP16 MMA - keep values in FP16, reinterpret to uint32_t
            half2 w0 = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
            half2 w1 = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 8 Q8 ELEMENTS (two FragB) for K=32 MMA
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(const int8_t* q8, FragB_t& frag0, FragB_t& frag1) {
        if constexpr (std::is_same_v<compute_t, half>) {
            frag0[0] = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
            frag0[1] = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
            frag1[0] = __halves2half2(__int2half_rn(q8[4]), __int2half_rn(q8[5]));
            frag1[1] = __halves2half2(__int2half_rn(q8[6]), __int2half_rn(q8[7]));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            frag0[0] = __halves2bfloat162(__int2bfloat16_rn(q8[0]), __int2bfloat16_rn(q8[1]));
            frag0[1] = __halves2bfloat162(__int2bfloat16_rn(q8[2]), __int2bfloat16_rn(q8[3]));
            frag1[0] = __halves2bfloat162(__int2bfloat16_rn(q8[4]), __int2bfloat16_rn(q8[5]));
            frag1[1] = __halves2bfloat162(__int2bfloat16_rn(q8[6]), __int2bfloat16_rn(q8[7]));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            half2 h01 = __halves2half2(__int2half_rn(q8[0]), __int2half_rn(q8[1]));
            half2 h23 = __halves2half2(__int2half_rn(q8[2]), __int2half_rn(q8[3]));
            half2 h45 = __halves2half2(__int2half_rn(q8[4]), __int2half_rn(q8[5]));
            half2 h67 = __halves2half2(__int2half_rn(q8[6]), __int2half_rn(q8[7]));
            
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
    // Q8_0 K/128 layout: 16 threads × 8 elements (8-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7 in int2 (8 bytes = 8 int8 values)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //
    // Q8_0 dequant: w = d * q8 where q8 is -128..127 (signed int8, symmetric)
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // PHASE 1: LANE-ONLY MATH (varies per thread) - 4 integer ops
        // =====================================================================
        const int row = lane >> 2;                              // SHR
        const int elem_off = (lane & 3) << 1;                   // LOP3 + SHL
        // row * 144 = row * 128 + row * 16 = (row << 7) + (row << 4)
        const uint8_t* row_base = smem_rows + (row << 7) + (row << 4);  // 2× SHL + 2× ADD
        
        // =====================================================================
        // PHASE 2: K_ITER LOOKUP (uniform across warp) - 3 LUT reads
        // Compiler will hoist and reuse across all 32 threads
        // =====================================================================
        constexpr uint8_t QS_LO[8] = {0, 16, 40, 56, 72, 88, 104, 128};
        constexpr uint8_t QS_HI[8] = {8, 32, 48, 64, 80, 96, 120, 136};
        constexpr uint8_t SCALE_OFF[4] = {24, 26, 112, 114};
        const int qs_lo = QS_LO[k_iter];
        const int qs_hi = QS_HI[k_iter];
        const int scale_off = SCALE_OFF[k_iter >> 1];
        
        // =====================================================================
        // PHASE 3: MEMORY LOADS - 3 ops, fully pipelined
        // =====================================================================
        const half d = *reinterpret_cast<const half*>(row_base + scale_off);
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(row_base + qs_lo + elem_off);
        const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + qs_hi + elem_off);
        
        // =====================================================================
        // PHASE 4: INT8→SCALED HALF2 via FLOAT32 INTERMEDIATE
        // Float path is faster: I2F.F32 + FMUL.F32 + CVT.F16X2 = 10 ops
        // vs half path: 4×(I2F+F2H) + 2×PRMT + 2×HMUL2 = 12 ops
        // =====================================================================
        const uint16_t lo = static_cast<uint16_t>(packed);
        
        if constexpr (std::is_same_v<compute_t, half>) {
            const float s = __half2float(d);
            // I2F fused with FMUL, then vectorized F32→F16x2
            frag[0] = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            frag[1] = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            const float s = __half2float(d);
            // Same pattern for bf16 - float intermediate is even better here
            // (avoids half→float→bf16→bf162 chain)
            frag[0] = __floats2bfloat162_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            frag[1] = __floats2bfloat162_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            const float s = __half2float(d);
            half2 r0 = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            half2 r1 = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
            frag[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag[1] = *reinterpret_cast<uint32_t*>(&r1);
        } else {
            // Fallback: compute in float, store as half2 (reinterpret for unknown types)
            const float s = __half2float(d);
            half2 r0 = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            half2 r1 = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
            frag[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag[1] = *reinterpret_cast<uint32_t*>(&r1);
        }
    }

    // -------------------------------------------------------------------------
    // 4x BATCHED DEQUANT - Calls optimized standalone function
    // -------------------------------------------------------------------------
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b)
    {
        dequant_q8_0_for_4x_mma_k16_runtime<compute_t, half_idx>(smem_rows, lane, frag_b);
    }
    
};

// Convenience aliases for Q8_0
using Q80_Dequant_FP16 = gemx_dequant_traits<block_c_q8_0, half, half>;
using Q80_Dequant_BF16 = gemx_dequant_traits<block_c_q8_0, __nv_bfloat16, __nv_bfloat16>;
using Q80_Dequant_FP8 = gemx_dequant_traits<block_c_q8_0, __nv_fp8_e4m3, half>;

// =============================================================================
// STANDALONE 4× K16 DEQUANT - Optimal balance of amortization vs register pressure
// =============================================================================
// Processes 4 k16 slices (half of K/128). Outputs 8 uint32_t fragments.
// Key optimizations:
//   - Lane math computed ONCE for all 4 slices (amortized 4×)
//   - Scale loaded as half2 pair (1 load for 2 scales)
//   - Grouped by scale for register efficiency
//   - Only 8 frag_b outputs live → better occupancy than x8
//
// half_idx: 0 = k16 slices 0-3 (d0|d1), 1 = k16 slices 4-7 (d2|d3)
// Output: frag_b[0..7] = 4 k16 slices × 2 frags each
// =============================================================================
template <typename compute_t, int half_idx>
__device__ __forceinline__ void dequant_q8_0_for_4x_mma_k16_runtime(
    const uint8_t* __restrict__ smem_rows,
    int lane,
    uint32_t* frag_b  // Output: 8 uint32_t (4 k16 slices × 2 frags each)
) {
    static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
    // =========================================================================
    // PHASE 1: LANE-ONLY MATH (computed ONCE for all 4 k16 slices)
    // Cost: 4 integer ops, amortized over 4 slices = 1.0 ops/slice
    // =========================================================================
    const int row = lane >> 2;
    const int elem_off = (lane & 3) << 1;
    const uint8_t* row_base = smem_rows + (row << 7) + (row << 4);  // row * 144
    
    // =========================================================================
    // PHASE 2: LOAD SCALE AS HALF2 PAIR
    // half_idx=0: offset 24 → d0|d1 for k16 0-3
    // half_idx=1: offset 112 → d2|d3 for k16 4-7
    // =========================================================================
    constexpr int scale_off = (half_idx == 0) ? 24 : 112;
    const half2 d_pair = *reinterpret_cast<const half2*>(row_base + scale_off);
    
    // =========================================================================
    // QS OFFSET TABLE for each half:
    // half_idx=0: k16 0: lo=0,hi=8   k16 1: lo=16,hi=32  k16 2: lo=40,hi=48  k16 3: lo=56,hi=64
    // half_idx=1: k16 4: lo=72,hi=80 k16 5: lo=88,hi=96  k16 6: lo=104,hi=120 k16 7: lo=128,hi=136
    // =========================================================================
    
    if constexpr (std::is_same_v<compute_t, half>) {
        // --- SCALE GROUP 0: d_lo for k16 slices 0-1 (or 4-5) → frag_b[0..3] ---
        {
            const half2 s2 = __half2half2(__low2half(d_pair));
            constexpr int base0 = (half_idx == 0) ? 0 : 72;
            constexpr int base1 = (half_idx == 0) ? 16 : 88;
            constexpr int hi0_off = (half_idx == 0) ? 8 : 80;
            constexpr int hi1_off = (half_idx == 0) ? 32 : 96;
            
            // k16 slice 0 (or 4)
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&r);
                }
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&r);
                }
            }
            
            // k16 slice 1 (or 5)
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&r);
                }
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&r);
                }
            }
        }
        
        // --- SCALE GROUP 1: d_hi for k16 slices 2-3 (or 6-7) → frag_b[4..7] ---
        {
            const half2 s2 = __half2half2(__high2half(d_pair));
            constexpr int base0 = (half_idx == 0) ? 40 : 104;
            constexpr int base1 = (half_idx == 0) ? 56 : 128;
            constexpr int hi0_off = (half_idx == 0) ? 48 : 120;
            constexpr int hi1_off = (half_idx == 0) ? 64 : 136;
            
            // k16 slice 2 (or 6)
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&r);
                }
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&r);
                }
            }
            
            // k16 slice 3 (or 7)
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&r);
                }
                {
                    half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                    half2 r = __hmul2(s2, q);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&r);
                }
            }
        }
        
    } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
        const float f_d0 = __half2float(__low2half(d_pair));
        const float f_d1 = __half2float(__high2half(d_pair));
        
        // --- SCALE GROUP 0 ---
        {
            const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(f_d0));
            constexpr int base0 = (half_idx == 0) ? 0 : 72;
            constexpr int base1 = (half_idx == 0) ? 16 : 88;
            constexpr int hi0_off = (half_idx == 0) ? 8 : 80;
            constexpr int hi1_off = (half_idx == 0) ? 32 : 96;
            
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)lo), __int2bfloat16_rn((int8_t)(lo >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[0] = *reinterpret_cast<uint32_t*>(&r); }
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)hi), __int2bfloat16_rn((int8_t)(hi >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[1] = *reinterpret_cast<uint32_t*>(&r); }
            }
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)lo), __int2bfloat16_rn((int8_t)(lo >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[2] = *reinterpret_cast<uint32_t*>(&r); }
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)hi), __int2bfloat16_rn((int8_t)(hi >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[3] = *reinterpret_cast<uint32_t*>(&r); }
            }
        }
        // --- SCALE GROUP 1 ---
        {
            const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(f_d1));
            constexpr int base0 = (half_idx == 0) ? 40 : 104;
            constexpr int base1 = (half_idx == 0) ? 56 : 128;
            constexpr int hi0_off = (half_idx == 0) ? 48 : 120;
            constexpr int hi1_off = (half_idx == 0) ? 64 : 136;
            
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)lo), __int2bfloat16_rn((int8_t)(lo >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[4] = *reinterpret_cast<uint32_t*>(&r); }
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)hi), __int2bfloat16_rn((int8_t)(hi >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[5] = *reinterpret_cast<uint32_t*>(&r); }
            }
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)lo), __int2bfloat16_rn((int8_t)(lo >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[6] = *reinterpret_cast<uint32_t*>(&r); }
                { __nv_bfloat162 q = __halves2bfloat162(__int2bfloat16_rn((int8_t)hi), __int2bfloat16_rn((int8_t)(hi >> 8)));
                  __nv_bfloat162 r = __hmul2(s2, q); frag_b[7] = *reinterpret_cast<uint32_t*>(&r); }
            }
        }
        
    } else {
        // FP8 or fallback: use half2
        // --- SCALE GROUP 0 ---
        {
            const half2 s2 = __half2half2(__low2half(d_pair));
            constexpr int base0 = (half_idx == 0) ? 0 : 72;
            constexpr int base1 = (half_idx == 0) ? 16 : 88;
            constexpr int hi0_off = (half_idx == 0) ? 8 : 80;
            constexpr int hi1_off = (half_idx == 0) ? 32 : 96;
            
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                { half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[0] = *reinterpret_cast<uint32_t*>(&r); }
                { half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[1] = *reinterpret_cast<uint32_t*>(&r); }
            }
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                { half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[2] = *reinterpret_cast<uint32_t*>(&r); }
                { half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[3] = *reinterpret_cast<uint32_t*>(&r); }
            }
        }
        // --- SCALE GROUP 1 ---
        {
            const half2 s2 = __half2half2(__high2half(d_pair));
            constexpr int base0 = (half_idx == 0) ? 40 : 104;
            constexpr int base1 = (half_idx == 0) ? 56 : 128;
            constexpr int hi0_off = (half_idx == 0) ? 48 : 120;
            constexpr int hi1_off = (half_idx == 0) ? 64 : 136;
            
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base0 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi0_off + elem_off);
                { half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[4] = *reinterpret_cast<uint32_t*>(&r); }
                { half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[5] = *reinterpret_cast<uint32_t*>(&r); }
            }
            {
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(row_base + base1 + elem_off);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + hi1_off + elem_off);
                { half2 q = __halves2half2(__int2half_rn((int8_t)lo), __int2half_rn((int8_t)(lo >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[6] = *reinterpret_cast<uint32_t*>(&r); }
                { half2 q = __halves2half2(__int2half_rn((int8_t)hi), __int2half_rn((int8_t)(hi >> 8)));
                  half2 r = __hmul2(s2, q); frag_b[7] = *reinterpret_cast<uint32_t*>(&r); }
            }
        }
    }
}
