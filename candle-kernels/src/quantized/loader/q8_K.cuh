#pragma once

// =============================================================================
// Q8_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// This loader processes Q8_K weights in K/128 format [K/128, N].
// Each block contains 128 × 8-bit signed quants plus embedded scale.
//
// Q8_K is 8-bit SYMMETRIC quantization: value = d * q8
// where q8 ∈ [-128, 127] (int8_t) and d is the scale.
// The original GGML Q8_K has float d and 256 elements per block,
// but for K/128 we split into 2 blocks of 128 elements, each with
// the same scale (converted to half).
//
// Q8_K is simpler than Q8_1 (no m/sum field needed for dequant).
// The dequantization formula is identical to Q8_0: value = d * q8
//
// LAYOUT (160 bytes, same structure as Q8_1)
// ------
// Weights: [K/128, N] block_c_q8_K_k128
//   - qs0..qs15 (int2): 16 × int8_t per 16-element slice
// Scales:  embedded half d for each 32-element quarter-block (all same value)
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
struct vec_dot_q_loader_q8_K_inline {
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
    // LOAD INTERFACE (K/128: single-part)
    // -------------------------------------------------------------------------
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q8_K* __restrict__ x,
        const int row,
        const int kbx,
        const int num_rows
    ) {
        static_assert(N == 0, "Q8_K uses 1-part interface for K/128");

        const int block_idx = kbx * num_rows + row;
        const block_c_q8_K_k128* __restrict__ blk = reinterpret_cast<const block_c_q8_K_k128*>(&x[block_idx]);

        // Q8_K K/128 layout (40 ints / 160 bytes) - same as Q8_1:
        // data[0-1]=qs0, data[2-3]=qs1, data[4-5]=qs2, data[6]=d0 (d|unused), data[7]=pad
        // data[8-9]=qs3, data[10-11]=qs4, data[12-13]=qs5, data[14-15]=qs6
        // data[16-17]=qs7, data[18-19]=qs8, data[20-21]=qs9, data[22-23]=qs10
        // data[24-25]=qs11, data[26-27]=qs12, data[28]=d1, data[29]=d2
        // data[30-31]=qs13, data[32-33]=qs14, data[34-35]=qs15, data[36]=d3, data[37-39]=pad
        //
        // All d values are the same (Q8_K has one scale per 256 elements).
        // Thread assignment to d groups (for layout compatibility):
        // Threads 0-3: d0 (data[6].x = d)
        // Threads 4-7: d1 (data[28].x = d)
        // Threads 8-11: d2 (data[29].x = d)
        // Threads 12-15: d3 (data[36].x = d)
        constexpr int qs_idx[16] = {0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 30, 32, 34};
        constexpr int d_idx[16] = {6, 6, 6, 6, 28, 28, 28, 28, 29, 29, 29, 29, 36, 36, 36, 36};

        const int lane = get_lane();
        
        // Load 8 int8_t quants as int2
        qs.x = blk->data[qs_idx[lane]];
        qs.y = blk->data[qs_idx[lane] + 1];

        // Load scale d from half2 (d.x = d, d.y = unused)
        const half2 d_pair = *reinterpret_cast<const half2*>(&blk->data[d_idx[lane]]);
        const half d_half = __low2half(d_pair);  // d is in low half
        d_x = to_acc<acc_t>(d_half);
    }
    
    // -------------------------------------------------------------------------
    // DOT PRODUCT - Single template, type-specialized internally (K/128: 8 elements)
    // Q8_K dequant is identical to Q8_0/Q8_1: value = d * q8
    // -------------------------------------------------------------------------
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N == 0, "Q8_K uses 1-part interface for K/128");
        
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
            
            // Convert scale to half (handles acc_t being FP8, float, or half)
            const half d_h = to_half(d_x);
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
        static_assert(N == 0, "Q8_K uses 1-part interface for K/128");
        
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
struct vec_dot_loader_for<block_q8_K, vdr, act_t> {
    using type = vec_dot_q_loader_q8_K_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q8_K (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q8_K, vdr, act_t> {
    using type = vec_dot_q_loader_q8_K_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// GEMX DEQUANT TRAITS - Q8_K (8-bit symmetric: value = d * q8)
// =============================================================================
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q8_K, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = false;  // Symmetric quantization
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q8_K>::scales_per_ktile;  // 1
    static constexpr int bits_per_element = 8;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q8_K K/128 layout (160 bytes = 40 ints):
    //   Each thread has 8 int8_t quants per K-tile
    //   Scales are half2 pairs (d, unused) embedded in the data
    //   All d values are the same for the entire 128-element block
    //
    // Layout (40 ints) - same structure as Q8_1:
    //   data[0-1]=qs0, data[2-3]=qs1, data[4-5]=qs2, data[6]=d0, data[7]=pad
    //   data[8-9]=qs3, data[10-11]=qs4, data[12-13]=qs5, data[14-15]=qs6
    //   data[16-17]=qs7, data[18-19]=qs8, data[20-21]=qs9, data[22-23]=qs10
    //   data[24-25]=qs11, data[26-27]=qs12, data[28]=d1, data[29]=d2
    //   data[30-31]=qs13, data[32-33]=qs14, data[34-35]=qs15, data[36]=d3
    //   data[37-39]=padding
    //
    // d values are all the same (one scale per 256 elements in original Q8_K)
    // =========================================================================
    
    static constexpr int K128_BYTES = 160;
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        const int row = lane >> 2;
        const int elem_off = (lane & 3) << 1;
        // row * 160 = row * 128 + row * 32 = (row << 7) + (row << 5)
        const uint8_t* row_base = smem_rows + (row << 7) + (row << 5);
        
        // Q8_K d offsets (same as Q8_1 dm offsets): d0=24, d1=112, d2=116, d3=144
        // For Q8_K, all d values are the same, so we could use any of them
        constexpr uint8_t D_OFF[4] = {24, 112, 116, 144};
        constexpr uint8_t QS_LO[8] = {0, 16, 40, 56, 72, 88, 104, 128};
        constexpr uint8_t QS_HI[8] = {8, 32, 48, 64, 80, 96, 120, 136};
        
        const int scale_group = k_iter >> 1;  // 0-3
        const int d_off = D_OFF[scale_group];
        const int qs_lo = QS_LO[k_iter];
        const int qs_hi = QS_HI[k_iter];
        
        // Load scale d from half2 pair (d.x = d, d.y = unused)
        const half2 d_pair = *reinterpret_cast<const half2*>(row_base + d_off);
        const half d = __low2half(d_pair);
        
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(row_base + qs_lo + elem_off);
        const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + qs_hi + elem_off);
        const uint16_t lo = static_cast<uint16_t>(packed);
        
        if constexpr (std::is_same_v<compute_t, half>) {
            const float s = __half2float(d);
            frag[0] = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            frag[1] = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            const float s = __half2float(d);
            frag[0] = __floats2bfloat162_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            frag[1] = __floats2bfloat162_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            const float s = __half2float(d);
            half2 r0 = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            half2 r1 = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
            frag[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag[1] = *reinterpret_cast<uint32_t*>(&r1);
        } else {
            const float s = __half2float(d);
            half2 r0 = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
            half2 r1 = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
            frag[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag[1] = *reinterpret_cast<uint32_t*>(&r1);
        }
    }

    // -------------------------------------------------------------------------
    // 4x BATCHED DEQUANT
    // -------------------------------------------------------------------------
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        constexpr int k_iter = half_idx * 4;  // 0 or 4
        
        const int row = lane >> 2;
        const int elem_off = (lane & 3) << 1;
        const uint8_t* row_base = smem_rows + (row << 7) + (row << 5);
        
        // Offsets for 4 k_iters: k_iter, k_iter+1, k_iter+2, k_iter+3
        constexpr uint8_t D_OFF[8] = {24, 24, 112, 112, 116, 116, 144, 144};
        constexpr uint8_t QS_LO[8] = {0, 16, 40, 56, 72, 88, 104, 128};
        constexpr uint8_t QS_HI[8] = {8, 32, 48, 64, 80, 96, 120, 136};
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int ki = k_iter + i;
            const int d_off = D_OFF[ki];
            const int qs_lo = QS_LO[ki];
            const int qs_hi = QS_HI[ki];
            
            const half2 d_pair = *reinterpret_cast<const half2*>(row_base + d_off);
            const half d = __low2half(d_pair);
            const float s = __half2float(d);
            
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(row_base + qs_lo + elem_off);
            const uint16_t hi = *reinterpret_cast<const uint16_t*>(row_base + qs_hi + elem_off);
            const uint16_t lo = static_cast<uint16_t>(packed);
            
            if constexpr (std::is_same_v<compute_t, half>) {
                frag_b[i * 2 + 0] = *reinterpret_cast<uint32_t*>(&__floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8))));
                frag_b[i * 2 + 1] = *reinterpret_cast<uint32_t*>(&__floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8))));
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                frag_b[i * 2 + 0] = *reinterpret_cast<uint32_t*>(&__floats2bfloat162_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8))));
                frag_b[i * 2 + 1] = *reinterpret_cast<uint32_t*>(&__floats2bfloat162_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8))));
            } else {
                half2 r0 = __floats2half2_rn(s * float((int8_t)lo), s * float((int8_t)(lo >> 8)));
                half2 r1 = __floats2half2_rn(s * float((int8_t)hi), s * float((int8_t)(hi >> 8)));
                frag_b[i * 2 + 0] = *reinterpret_cast<uint32_t*>(&r0);
                frag_b[i * 2 + 1] = *reinterpret_cast<uint32_t*>(&r1);
            }
        }
    }
};
