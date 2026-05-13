#pragma once

// =============================================================================
// Q4_K LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// This loader processes Q4_K weights in K/128 format [K/128, N].
// Each block contains 128 × 4-bit weights plus embedded half2 scales.
//
// LAYOUT
// ------
// Weights: [K/128, N] block_c_q4_K_k128
// Scales:  embedded half2[4] per 32 elements (4 groups of 32)
//
// THREAD MAPPING & WARP UTILIZATION
// ---------------------------------
// Each K/128 block requires 16 threads (each thread handles 8 elements).
// The kernel runs with 128 threads (4 warps), processing 8 K-blocks in parallel:
//
//   Threads 0-15   → K-block 0, lane = threadIdx.x & 15 → 0-15
//   Threads 16-31  → K-block 1, lane = threadIdx.x & 15 → 0-15
//   Threads 32-47  → K-block 2, lane = threadIdx.x & 15 → 0-15
//   ...            → ...       ...
//   Threads 112-127→ K-block 7, lane = threadIdx.x & 15 → 0-15
//
// IMPORTANT: get_lane() = threadIdx.x & 15 does NOT mean 50% warp utilization!
// All 128 threads are active - the masking is just for indexing within each
// 16-thread group that handles one K/128 block. This achieves 100% utilization.
//
// OPTIMIZATIONS
// -------------
// - Scales stored as half2: no conversion for half/bf16 paths
// - Batch nibble extraction: separate lo/hi nibbles, then byte extraction
// - float4 vector loads/stores: 4× fewer memory transactions
// - half2/bf162 arithmetic: 2× throughput for FP16/BF16 Y inputs
// - Register-pressure optimized: process in groups, immediate consumption
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// =============================================================================
// K-TILE LOADER FOR Q4_K
// =============================================================================

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q4_K_inline {
    // -------------------------------------------------------------------------
    // TYPE ALIASES
    // -------------------------------------------------------------------------
    using acc2_type = acc2_for_act_t<acc_t>;
    
    // -------------------------------------------------------------------------
    // STRUCT FIELDS - scales stored in native acc2 format
    // f32 → float2, f16 → half2, bf16 → bfloat162
    // -------------------------------------------------------------------------
    int v;                   // 4B - 8 × 4-bit weights as single int
    acc2_type sm;            // (scale, neg_min) in native format for acc_t
    
    // -------------------------------------------------------------------------
    // COMPUTE THREAD-SPECIFIC INDICES
    // -------------------------------------------------------------------------
    // get_lane() returns 0-15 for indexing within a 16-thread group.
    // This does NOT mean half the threads are idle! The kernel processes
    // 8 K-blocks in parallel: threads 0-15 handle block 0, 16-31 handle block 1, etc.
    // All 128 threads are active (100% warp utilization).
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;  // Lane within 16-thread group (NOT warp lane!)
    }
    
    // -------------------------------------------------------------------------
    // 4-PART LOAD INTERFACE (16 elements per part)
    // -------------------------------------------------------------------------

    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q4_K* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q4_K uses 16-part interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q4_K_k128* __restrict__ blk = reinterpret_cast<const block_c_q4_K_k128*>(&x[block_idx]);

        // K/128: 16 threads per block, each loads 1 int (8 elements).
        // Threads 0-3 use dm0, threads 4-7 use dm1, threads 8-11 use dm2, threads 12-15 use dm3.
        const int lane = get_lane();
        
        // Access qs0-qs15 via data[] array (qs fields are at specific offsets)
        // Block layout: qs0-3, dm0, dm1, qs4-7, qs8-11, dm2, dm3, qs12-15
        // data[] indices: 0-3=qs0-3, 4=dm0, 5=dm1, 6-9=qs4-7, 10-13=qs8-11, 14=dm2, 15=dm3, 16-19=qs12-15
        static constexpr int qs_idx[16] = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19};
        v = blk->data[qs_idx[lane]];
        
        // Load scales: dm0 for lanes 0-3, dm1 for 4-7, dm2 for 8-11, dm3 for 12-15
        // dm0=data[4], dm1=data[5], dm2=data[14], dm3=data[15]
        static constexpr int dm_idx[4] = {4, 5, 14, 15};
        const half2* dm_ptr = reinterpret_cast<const half2*>(&blk->data[dm_idx[lane >> 2]]);
        sm = convert_half2_to_acc2<acc2_type>(*dm_ptr);
    }
    
    // -------------------------------------------------------------------------
    // DOT PRODUCT - Single template, type-specialized internally
    // K/128: 8 elements per thread (4 iterations of half2/bfloat162)
    // -------------------------------------------------------------------------
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q4_K uses 16-part interface (K/128)");
        
        // =================================================================
        // LOP3-READY NIBBLE EXTRACTION
        // =================================================================
        // The weights are packed with LOP3-ready layout for fast extraction.
        // Instead of lo = v & 0x0F0F0F0F; hi = (v >> 4) & 0x0F0F0F0F; + 4× PRMT,
        // we use direct shifts that produce half2-aligned pairs:
        //
        // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
        //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
        //
        // Extraction (with LO_MASK = 0x000f000f extracting bits[3:0] and bits[19:16]):
        //   v        → (n0, n1) pair
        //   v >> 4   → (n4, n5) pair  
        //   v >> 8   → (n2, n3) pair
        //   v >> 12  → (n6, n7) pair
        //
        // This eliminates 4 PRMT + 2 AND + 1 SHIFT instructions.
        // =================================================================
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH: 8 elements per thread
            // For float, we extract nibbles individually since we need float precision
            const float scale = to_f32(lo_acc2(sm));
            const float neg_min = to_f32(hi_acc2(sm));
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // Extract nibbles from LOP3-ready layout
            // n0=bits[3:0], n1=bits[19:16], n2=bits[11:8], n3=bits[27:24]
            // n4=bits[7:4], n5=bits[23:20], n6=bits[15:12], n7=bits[31:28]
            const int n0 = v & 0xF;
            const int n1 = (v >> 16) & 0xF;
            const int n2 = (v >> 8) & 0xF;
            const int n3 = (v >> 24) & 0xF;
            const int n4 = (v >> 4) & 0xF;
            const int n5 = (v >> 20) & 0xF;
            const int n6 = (v >> 12) & 0xF;
            const int n7 = (v >> 28) & 0xF;
            
            float sum;
            {
                const float4 yv = y4[0];
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n0), neg_min), yv.x, 0.0f);   // n0 * y[0]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n1), neg_min), yv.y, sum);    // n1 * y[1]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n2), neg_min), yv.z, sum);    // n2 * y[2]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n3), neg_min), yv.w, sum);    // n3 * y[3]
            }
            {
                const float4 yv = y4[1];
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n4), neg_min), yv.x, sum);    // n4 * y[4]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n5), neg_min), yv.y, sum);    // n5 * y[5]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n6), neg_min), yv.z, sum);    // n6 * y[6]
                sum  = __fmaf_rn(__fmaf_rn(scale, float(n7), neg_min), yv.w, sum);    // n7 * y[7]
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH: 8 elements = 4 half2 iterations
            // LOP3 OPTIMIZATION with shift-based extraction (no PRMT needed)
            const half2 scale2 = __half2half2(lo_acc2(sm));
            const half2 neg_min2 = __half2half2(hi_acc2(sm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // LOP3 magic constants for FP16:
            // EX = 0x64006400 is exp=10 (bias 15), so mantissa bits become value+1024
            // BIAS = 0x64006400 = half2(1024, 1024)
            // LO_MASK = 0x000f000f extracts lo nibble from lo/hi half words
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;  // half2(1024, 1024)
            constexpr int LO_MASK = 0x000f000f;
            
            // SHIFT-BASED EXTRACTION: Direct shifts to get half2-aligned pairs
            //   v        → (n0, n1) pair → y2[0] = (y[0], y[1])
            //   v >> 8   → (n2, n3) pair → y2[1] = (y[2], y[3])
            //   v >> 4   → (n4, n5) pair → y2[2] = (y[4], y[5])
            //   v >> 12  → (n6, n7) pair → y2[3] = (y[6], y[7])
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)v, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 8), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 4), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 12), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH: 8 elements = 4 bfloat162 iterations
            // LOP3 OPTIMIZATION with shift-based extraction (no PRMT needed)
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(lo_acc2(sm));
            const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(hi_acc2(sm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // LOP3 magic constants for BF16:
            // EX = 0x43004300 is exp=134-127=7 (bias 127), so we get 128+nibble
            // BIAS = 0x43004300 = bf162(128, 128)
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;  // bf162(128, 128)
            constexpr int LO_MASK = 0x000f000f;
            
            // SHIFT-BASED EXTRACTION: Direct shifts to get bf162-aligned pairs
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)v, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw), 
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 8), LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw), 
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 4), LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw), 
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 12), LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw), 
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[3], sum2);
            }
            
            return __bfloat162float(__low2bfloat16(sum2)) + __bfloat162float(__high2bfloat16(sum2));
            
        } else {
            // =================================================================
            // FP8 SPECIALIZED PATH - High performance via FP16 accumulation
            // =================================================================
            // This else branch handles FP8 (y_t == __nv_fp8_e4m3).
            // We use a static_assert to ensure no unexpected types reach here.
            // Note: We cannot use std::is_same_v<y_t, __nv_fp8_e4m3> in an
            // "else if constexpr" because MSVC's host parser chokes on it.
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half scale_h = lo_acc2(sm);
            const half neg_min_h = hi_acc2(sm);
            const half2 scale2 = __half2half2(scale_h);
            const half2 neg_min2 = __half2half2(neg_min_h);
            
            // Load FP8 inputs as uint32_t (4 FP8 values per load = 8 values in 2 loads)
            const uint32_t* y_u32 = reinterpret_cast<const uint32_t*>(y + get_lane() * 8);
            const uint32_t y_packed0 = y_u32[0];  // y[0..3] as FP8
            const uint32_t y_packed1 = y_u32[1];  // y[4..7] as FP8
            
            // Convert FP8x2 pairs to half2 using hardware conversion (SM89+)
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            // Hardware FP8→FP16 conversion: __nv_cvt_fp8x2_to_halfraw2 + cast
            const __nv_fp8x2_storage_t* fp8_ptr0 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed0);
            const __nv_fp8x2_storage_t* fp8_ptr1 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed1);
            
            __half2_raw y0_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[0], __NV_E4M3);  // y[0..1]
            __half2_raw y1_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[1], __NV_E4M3);  // y[2..3]
            __half2_raw y2_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[0], __NV_E4M3);  // y[4..5]
            __half2_raw y3_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[1], __NV_E4M3);  // y[6..7]
            
            const half2 y0 = *reinterpret_cast<half2*>(&y0_raw);
            const half2 y1 = *reinterpret_cast<half2*>(&y1_raw);
            const half2 y2_v = *reinterpret_cast<half2*>(&y2_raw);
            const half2 y3 = *reinterpret_cast<half2*>(&y3_raw);
            #else
            // Software conversion for SM80-SM88: use precomputed 256-entry LUT
            // Reduces ~15 ALU instructions per byte to a single constant memory load
            // LUT is in math.cuh: fp8_e4m3_to_half_lut[256]
            half2 y0, y1, y2_v, y3;
            fp8x4_to_half2x2_lut(y_packed0, y0, y1);
            fp8x4_to_half2x2_lut(y_packed1, y2_v, y3);
            #endif
            
            // LOP3 weight dequantization with SHIFT-BASED EXTRACTION (no PRMT needed)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x000f000f;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)v, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y0, sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 8), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y1, sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 4), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2_v, sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(v >> 12), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y3, sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    // -------------------------------------------------------------------------
    // DEQUANTIZE - K/128: 8 elements per thread = 2 float4 stores
    // -------------------------------------------------------------------------
    
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N < 16, "Q4_K uses 16-part interface (K/128)");
        
        const float scale = to_f32(lo_acc2(sm));
        const float neg_min = to_f32(hi_acc2(sm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        
        // Extract 8 nibbles from LOP3-ready packed layout
        // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
        //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
        const int n0 = v & 0xF;
        const int n1 = (v >> 16) & 0xF;
        const int n2 = (v >> 8) & 0xF;
        const int n3 = (v >> 24) & 0xF;
        const int n4 = (v >> 4) & 0xF;
        const int n5 = (v >> 20) & 0xF;
        const int n6 = (v >> 12) & 0xF;
        const int n7 = (v >> 28) & 0xF;
        
        // Output in sequential order: out[0..7] = dequant(n0, n1, n2, n3, n4, n5, n6, n7)
        out4[0] = make_float4(
            __fmaf_rn(scale, float(n0), neg_min),  // n0
            __fmaf_rn(scale, float(n1), neg_min),  // n1
            __fmaf_rn(scale, float(n2), neg_min),  // n2
            __fmaf_rn(scale, float(n3), neg_min)   // n3
        );
        
        out4[1] = make_float4(
            __fmaf_rn(scale, float(n4), neg_min),  // n4
            __fmaf_rn(scale, float(n5), neg_min),  // n5
            __fmaf_rn(scale, float(n6), neg_min),  // n6
            __fmaf_rn(scale, float(n7), neg_min)   // n7
        );
    }
};

// =============================================================================
// TRAIT SPECIALIZATIONS
// =============================================================================
// Maps act_t → acc_t using: f32→f32, f16→f16, bf16→bf16, fp8→f16

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q4_K, vdr, act_t> {
    using type = vec_dot_q_loader_q4_K_inline<vdr, acc_for_act_t<act_t>>;
};

// K/128 format alias - same loader, different block type
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q4_K, vdr, act_t> {
    using type = vec_dot_q_loader_q4_K_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// GEMX DEQUANT TRAITS - Q4_K (4-bit K-quant with scale and min)
// =============================================================================
// Include the base gemx_dequant infrastructure
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q4_K, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q4_K>::scales_per_ktile;  // 1
    static constexpr int bits_per_element = 4;
    
    // Fragment types (self-contained)
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // MMA LANE-DISPATCHED DEQUANTIZATION - FULLY COMPILE-TIME
    // =========================================================================
    //
    // MMA m16n8k16 thread layout: 32 threads map to 8 rows × 4 K-groups
    //   Lane 0-3:   Row 0, K-groups 0-3 (K positions 0-15)
    //   Lane 4-7:   Row 1, K-groups 0-3
    //   ...
    //   Lane 28-31: Row 7, K-groups 0-3
    //
    // Each thread provides 4 consecutive K-elements as a FragB.
    // With lane dispatch, N (row) and K_GROUP are compile-time constants.
    //
    // Q4_K K/128 layout (80 bytes):
    //   qs0-qs3 (16B) + dm0 (4B) + dm1 (4B) + qs4-qs7 (16B) +
    //   qs8-qs11 (16B) + dm2 (4B) + dm3 (4B) + qs12-qs15 (16B)
    //
    // Each qsN is 4 bytes = 8 nibbles (8 elements in LOP3-ready order).
    // Scale groups: dm0 covers qs0-3 (elem 0-31), dm1 covers qs4-7 (elem 32-63),
    //               dm2 covers qs8-11 (elem 64-95), dm3 covers qs12-15 (elem 96-127)
    //
    // =========================================================================
    
    // K/128 block byte offsets (compile-time constants)
    static constexpr int K128_BYTES = 80;
    static constexpr int QS_STRIDE = 4;  // 4 bytes per thread's qs
    
    // Thread-to-qs mapping: which qs field does MMA lane's row correspond to?
    // Lane's N (0-7) maps to qs{N*2} and qs{N*2+1} for the two halves of the row's data
    // But we need K/128 data for 8 consecutive rows, and each row has one K/128 block
    
    // -------------------------------------------------------------------------
    // COMPILE-TIME LANE PARAMETERS
    // -------------------------------------------------------------------------
    
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;           // Row index (0-7)
        static constexpr int K_GROUP = LANE % 4;     // Which 4-element group within K=16 (0-3)
        
        // Each K/128 block: 16 threads × 8 elements = 128 elements
        // For MMA, we need 4 elements from K-position K_GROUP*4 to K_GROUP*4+3
        // Element i in K/128 block → thread (i/8), nibble position (i%8) within thread's int
        //
        // K_GROUP 0: elements 0-3   → thread 0, nibbles 0-3
        // K_GROUP 1: elements 4-7   → thread 0, nibbles 4-7
        // K_GROUP 2: elements 8-11  → thread 1, nibbles 0-3
        // K_GROUP 3: elements 12-15 → thread 1, nibbles 4-7
        
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
    // Extracts all 8 nibbles from one int into two FragB
    // Used for FP8 MMA m16n8k32 where each thread provides 8 K-elements
    
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
    // DEQUANT FOR MMA K=16 (FP16/BF16 MMA m16n8k16)
    // =========================================================================
    // K_ITER: which K/16 tile within K/128 (0-7)
    // LANE: warp lane (0-31), determines N and K_GROUP at compile time
    // smem_rows: 8 consecutive K/128 blocks in shared memory
    // frag: output FragB (4 elements)
    // scale, neg_min: output scale parameters
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA m16n8k16
    // =========================================================================
    // For MMA m16n8k16, thread t in warp provides B[k, n] where:
    //   n = t / 4 (row 0-7)
    //   k positions: {(t%4)*2, (t%4)*2+1, (t%4)*2+8, (t%4)*2+9}
    //
    // Fragment layout:
    //   frag[0] = half2(B[k0, n], B[k1, n]) where k0=(t%4)*2, k1=k0+1
    //   frag[1] = half2(B[k0+8, n], B[k1+8, n])
    //
    // K/128 data layout with interleaving:
    //   qs[i] contains elements i*8 to i*8+7, but interleaved:
    //   nibble positions: elem 0,1,2,3,4,5,6,7 → nibbles 0,2,4,6,1,3,5,7
    //
    // For k_iter (which K/16 slice):
    //   qs[k_iter*2]   has K = k_iter*16 + {0..7}
    //   qs[k_iter*2+1] has K = k_iter*16 + {8..15}
    //
    // extract_4_elements<0> gives: frag[0]=half2(e0,e2), frag[1]=half2(e1,e3)
    // extract_4_elements<1> gives: frag[0]=half2(e4,e6), frag[1]=half2(e5,e7)
    //
    // MMA needs for k_group=0: half2(e0,e1) from qs_lo, half2(e8,e9) from qs_hi
    // So we extract from both qs, then repack using __lows2half2/__highs2half2
    // =========================================================================
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // OPTIMIZED ADDRESS COMPUTATION - All bit operations, no divisions
        // =====================================================================
        // Lane decomposition: lane = row*4 + k_group
        const int row = lane >> 2;          // N: 0-7 (which row)
        const int k_group = lane & 3;       // 0-3 (which K pair)
        
        // Base pointer for this row's K/128 block
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // qs offset formula: gaps at bytes 16-23 (after qs3) and 56-63 (after qs11)
        // qs_offset = k_iter*8 + 8*(k_iter>=2) + 8*(k_iter>=6)
        const int qs_offset = (k_iter + (k_iter >= 2) + (k_iter >= 6)) << 3;
        
        // dm offset formula: dm0=16, dm1=20, dm2=56, dm3=60
        // dm_idx = k_iter/2, but we can compute directly from k_iter:
        // dm_offset = 16 + (k_iter>=4)*40 + ((k_iter&2)<<1)
        const int dm_offset = 16 + ((k_iter >= 4) * 40) + ((k_iter & 2) << 1);
        
        // VECTORIZED LOAD: Load both qs ints as int2
        const int2 qs_pair = *reinterpret_cast<const int2*>(row_base + qs_offset);
        const int qs_lo = qs_pair.x;
        const int qs_hi = qs_pair.y;
        
        // Load scale/min (half2)
        const half2 dm = *reinterpret_cast<const half2*>(row_base + dm_offset);
        
        // =====================================================================
        // OPTIMIZED SHIFT COMPUTATION - Pure bit operations
        // =====================================================================
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // Formula: shift = ((k_group & 1) << 3) | ((k_group & 2) << 1)
        //   k_group=0: (0<<3)|(0<<1) = 0  ✓
        //   k_group=1: (1<<3)|(0<<1) = 8  ✓
        //   k_group=2: (0<<3)|(2<<1) = 4  ✓
        //   k_group=3: (1<<3)|(2<<1) = 12 ✓
        constexpr uint32_t LO_MASK = 0x000f000f;
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            const half2 scale2 = __half2half2(__low2half(dm));
            const half2 neg_min2 = __half2half2(__high2half(dm));
            
            // Extract ONLY the pair we need from qs_lo and qs_hi
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
            
            // Apply scale and neg_min: w = scale * raw + neg_min
            frag[0] = __hfma2(scale2, raw0, neg_min2);
            frag[1] = __hfma2(scale2, raw1, neg_min2);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t BIAS = 0x43004300;
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm))));
            const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm))));
            
            // Extract ONLY the pair we need from qs_lo and qs_hi
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
            
            // Apply scale and neg_min: w = scale * raw + neg_min
            frag[0] = __hfma2(scale2, raw0, neg_min2);
            frag[1] = __hfma2(scale2, raw1, neg_min2);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            // FP8 compute type uses FP16 MMA (dequant outputs FP16-packed fragments)
            // Same as half path - keep values in FP16 for MMA compatibility
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            const half2 scale2 = __half2half2(__low2half(dm));
            const half2 neg_min2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
            
            // Apply scale and neg_min in FP16, keep as FP16 for MMA
            half2 w0 = __hfma2(scale2, raw0, neg_min2);
            half2 w1 = __hfma2(scale2, raw1, neg_min2);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR MMA m16n8k32 (FP8 only) - Processes K=32 per iteration
    // =========================================================================
    // For MMA m16n8k32, each thread provides 8 B elements as 2×uint32 (4 FP8 each):
    //   b[0]: K positions {(lane%4)*4, +1, +2, +3}        - contiguous 4
    //   b[1]: K positions {(lane%4)*4+16, +17, +18, +19}  - contiguous 4 at K+16
    //   n = lane/4 (row 0-7)
    //
    // k_iter: which K/32 slice within K/128 (0-3)
    // lane: warp lane (0-31)
    // frag_b: output 2×uint32 with 8 FP8 values total
    //
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY UNROLLED)
    // =========================================================================
    // Optimizations (following Q8_0 pattern):
    // 1. Lane math computed ONCE for all 4 slices
    // 2. All qs loads HOISTED upfront (ILP / memory latency hiding)
    // 3. Scales loaded together
    // 4. Grouped by scale (2 slices per scale)
    // 5. Explicit offset tables - no runtime computation
    // 6. Shift precomputed once
    // 7. half_idx as template parameter (compile-time branch elimination)
    //
    // Q4_K K/128 layout (80 bytes):
    //   qs0-3 at 0-15, dm0 at 16, dm1 at 20, qs4-7 at 24-39,
    //   qs8-11 at 40-55, dm2 at 56, dm3 at 60, qs12-15 at 64-79
    //
    // k_iter → qs_offset mapping:
    //   0→0, 1→8, 2→24, 3→32, 4→40, 5→48, 6→64, 7→72
    //
    // Scale grouping:
    //   half_idx=0: dm0 at 16 (k_iter 0,1), dm1 at 20 (k_iter 2,3)
    //   half_idx=1: dm2 at 56 (k_iter 4,5), dm3 at 60 (k_iter 6,7)
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
        // Load scales - compile-time constant offsets
        constexpr int dm_base = (half_idx == 0) ? 16 : 56;
        const half2 dm0 = *reinterpret_cast<const half2*>(row_base + dm_base);
        const half2 dm1 = *reinterpret_cast<const half2*>(row_base + dm_base + 4);
        
        // Load all 4 qs pairs upfront - compile-time constant offsets
        // half_idx=0: offsets 0, 8, 24, 32
        // half_idx=1: offsets 40, 48, 64, 72
        constexpr int qs0_off = (half_idx == 0) ? 0 : 40;
        constexpr int qs1_off = (half_idx == 0) ? 8 : 48;
        constexpr int qs2_off = (half_idx == 0) ? 24 : 64;
        constexpr int qs3_off = (half_idx == 0) ? 32 : 72;
        
        const int2 qs0 = *reinterpret_cast<const int2*>(row_base + qs0_off);
        const int2 qs1 = *reinterpret_cast<const int2*>(row_base + qs1_off);
        const int2 qs2 = *reinterpret_cast<const int2*>(row_base + qs2_off);
        const int2 qs3 = *reinterpret_cast<const int2*>(row_base + qs3_off);
        
        // =====================================================================
        // PHASE 3: TYPE-SPECIFIC DEQUANTIZATION
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            
            // --- SCALE GROUP 0: dm0 for k_iter 0,1 (or 4,5) → frag_b[0..3] ---
            {
                const half2 scale2 = __half2half2(__low2half(dm0));
                const half2 neg_min2 = __half2half2(__high2half(dm0));
                
                // k_iter 0 (or 4)
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1 (or 5)
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: dm1 for k_iter 2,3 (or 6,7) → frag_b[4..7] ---
            {
                const half2 scale2 = __half2half2(__low2half(dm1));
                const half2 neg_min2 = __half2half2(__high2half(dm1));
                
                // k_iter 2 (or 6)
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3 (or 7)
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t BIAS = 0x43004300;
            
            // Convert scales once
            const float f_scale0 = __half2float(__low2half(dm0));
            const float f_neg_min0 = __half2float(__high2half(dm0));
            const float f_scale1 = __half2float(__low2half(dm1));
            const float f_neg_min1 = __half2float(__high2half(dm1));
            
            // --- SCALE GROUP 0 ---
            {
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(f_scale0));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(f_neg_min0));
                
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                    __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                    __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1 ---
            {
                const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(f_scale1));
                const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(f_neg_min1));
                
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                    __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                    __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_min2);
                    __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else {
            // FP8 path (uses FP16 dequant)
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            
            // --- SCALE GROUP 0 ---
            {
                const half2 scale2 = __half2half2(__low2half(dm0));
                const half2 neg_min2 = __half2half2(__high2half(dm0));
                
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1 ---
            {
                const half2 scale2 = __half2half2(__low2half(dm1));
                const half2 neg_min2 = __half2half2(__high2half(dm1));
                
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                    half2 w0 = __hfma2(scale2, raw0, neg_min2);
                    half2 w1 = __hfma2(scale2, raw1, neg_min2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
        }
    }
};



