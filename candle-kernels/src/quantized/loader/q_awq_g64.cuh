#pragma once

// =============================================================================
// Q_AWQ_G64 LOADER - AWQ WITH GROUP SIZE 64 IN K/128 FORMAT
// =============================================================================
//
// This loader processes AWQ weights with group size 64 (2 groups per K/128).
// Uses 4-bit asymmetric quantization with per-group scales and zeros.
// Dequantization: w = scale * (q - zero) = scale * q + neg_sz
//   where neg_sz = -scale * zero (precomputed for FMA efficiency)
//
// LAYOUT
// ------
// Weights: [K/128, N] block_c_q_awq_g64_k128 (80 bytes per block)
//   - qs[16]: int32 containing packed 4-bit weights (8 weights per int32)
//   - scales[2]: half for each group of 64
//   - zeros[2]: half zero points for each group
//   - _pad: 8 bytes padding for 16-byte alignment
//
// THREAD MAPPING
// --------------
// 16 threads per K/128 block, each thread handles 8 elements.
// Threads 0-7 use group 0 (first 64 elements, k_iter 0-3)
// Threads 8-15 use group 1 (second 64 elements, k_iter 4-7)
//
// =============================================================================

#include "../impl/common.cuh"
#include "../block_compact.cuh"
#include "../math.cuh"
#include "scale_types.cuh"
#include "gemx_dequant.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// =============================================================================
// VEC_DOT LOADER FOR AWQ G64 K/128 FORMAT
// =============================================================================

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q_awq_g64_inline {
    // -------------------------------------------------------------------------
    // STRUCT FIELDS - Using acc2_type pattern for (scale, neg_sz)
    // -------------------------------------------------------------------------
    uint32_t q_packed;       // 8 × 4-bit weights
    using acc2_type = half2; // (scale, neg_sz) packed
    acc2_type sm;            // sm.x = scale, sm.y = neg_sz = -scale * zero
    
    // -------------------------------------------------------------------------
    // THREAD INDEX HELPERS
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;  // 0-15 for K/128
    }
    
    __device__ __forceinline__ static int get_group() {
        return (threadIdx.x & 15) >> 3;  // 0 for lanes 0-7, 1 for lanes 8-15
    }
    
    // -------------------------------------------------------------------------
    // LOAD INTERFACE - Precomputes neg_sz for FMA efficiency
    // -------------------------------------------------------------------------
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q_awq_g64* __restrict__ x,
        const int row,
        const int kbx,
        const int num_rows
    ) {
        static_assert(N == 0, "Q_AWQ_G64 uses 1-part interface for K/128");

        const int block_idx = kbx * num_rows + row;
        const block_c_q_awq_g64_k128* __restrict__ blk = reinterpret_cast<const block_c_q_awq_g64_k128*>(&x[block_idx]);

        const int lane = get_lane();
        const int group = get_group();
        
        q_packed = blk->qs[lane];
        
        // Precompute neg_sz = -scale * zero for FMA: w = scale * q + neg_sz
        const half scale = blk->scales[group];
        const half zero = blk->zeros[group];
        const half neg_sz = __hmul(__hneg(scale), zero);
        sm = make_half2(scale, neg_sz);
    }
    
    // -------------------------------------------------------------------------
    // DOT PRODUCT: w = scale * q + neg_sz (FMA form)
    // -------------------------------------------------------------------------
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N == 0, "Q_AWQ_G64 uses 1-part interface for K/128");
        
        // Extract scale and neg_sz
        const float scale = __half2float(__low2half(sm));
        const float neg_sz = __half2float(__high2half(sm));
        
        // LOP3-based nibble extraction using same pattern as Q4_K
        // Nibble layout in q_packed: n0|n1|n2|n3|n4|n5|n6|n7 (each 4 bits)
        // Shifts: 0→(n0,n1), 8→(n2,n3), 4→(n4,n5), 12→(n6,n7)
        constexpr uint32_t LO_MASK = 0x000f000f;
        constexpr uint32_t EX = 0x64006400;
        constexpr uint32_t BIAS = 0x64006400;
        
        // Extract 4 pairs of nibbles
        int raw01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 0), LO_MASK, EX);
        int raw23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 8), LO_MASK, EX);
        int raw45 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 4), LO_MASK, EX);
        int raw67 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 12), LO_MASK, EX);
        
        // Convert to half2 and subtract bias
        half2 h01 = __hsub2(*reinterpret_cast<half2*>(&raw01), *reinterpret_cast<const half2*>(&BIAS));
        half2 h23 = __hsub2(*reinterpret_cast<half2*>(&raw23), *reinterpret_cast<const half2*>(&BIAS));
        half2 h45 = __hsub2(*reinterpret_cast<half2*>(&raw45), *reinterpret_cast<const half2*>(&BIAS));
        half2 h67 = __hsub2(*reinterpret_cast<half2*>(&raw67), *reinterpret_cast<const half2*>(&BIAS));
        
        // AWQ dequant: w = scale * q + neg_sz
        const half2 scale2 = __half2half2(__low2half(sm));
        const half2 neg_sz2 = __half2half2(__high2half(sm));
        
        half2 w01 = __hfma2(scale2, h01, neg_sz2);
        half2 w23 = __hfma2(scale2, h23, neg_sz2);
        half2 w45 = __hfma2(scale2, h45, neg_sz2);
        half2 w67 = __hfma2(scale2, h67, neg_sz2);
        
        // Convert to floats for accumulation (match element order: 0,1,2,3,4,5,6,7)
        float w0 = __half2float(__low2half(w01));
        float w1 = __half2float(__high2half(w01));
        float w2 = __half2float(__low2half(w23));
        float w3 = __half2float(__high2half(w23));
        float w4 = __half2float(__low2half(w45));
        float w5 = __half2float(__high2half(w45));
        float w6 = __half2float(__low2half(w67));
        float w7 = __half2float(__high2half(w67));
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            float sum = 0.0f;
            {
                const float4 yv = y4[0];
                sum = __fmaf_rn(w0, yv.x, sum);
                sum = __fmaf_rn(w1, yv.y, sum);
                sum = __fmaf_rn(w2, yv.z, sum);
                sum = __fmaf_rn(w3, yv.w, sum);
            }
            {
                const float4 yv = y4[1];
                sum = __fmaf_rn(w4, yv.x, sum);
                sum = __fmaf_rn(w5, yv.y, sum);
                sum = __fmaf_rn(w6, yv.z, sum);
                sum = __fmaf_rn(w7, yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH - use pre-dequantized w values
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            half2 sum2 = __float2half2_rn(0.0f);
            sum2 = __hfma2(w01, y2[0], sum2);
            sum2 = __hfma2(w23, y2[1], sum2);
            sum2 = __hfma2(w45, y2[2], sum2);
            sum2 = __hfma2(w67, y2[3], sum2);
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            __nv_bfloat162 wb01 = __floats2bfloat162_rn(w0, w1);
            __nv_bfloat162 wb23 = __floats2bfloat162_rn(w2, w3);
            __nv_bfloat162 wb45 = __floats2bfloat162_rn(w4, w5);
            __nv_bfloat162 wb67 = __floats2bfloat162_rn(w6, w7);
            
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            sum2 = __hfma2(wb01, y2[0], sum2);
            sum2 = __hfma2(wb23, y2[1], sum2);
            sum2 = __hfma2(wb45, y2[2], sum2);
            sum2 = __hfma2(wb67, y2[3], sum2);
            
            return __bfloat162float(__low2bfloat16(sum2)) + __bfloat162float(__high2bfloat16(sum2));
            
        } else {
            // FP8 PATH
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y");
            
            const uint32_t* y_u32 = reinterpret_cast<const uint32_t*>(y + get_lane() * 8);
            const uint32_t y_packed0 = y_u32[0];
            const uint32_t y_packed1 = y_u32[1];
            
            half2 yh0, yh1, yh2, yh3;
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            const __nv_fp8x2_storage_t* fp8_ptr0 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed0);
            const __nv_fp8x2_storage_t* fp8_ptr1 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&y_packed1);
            
            __half2_raw y0_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[0], __NV_E4M3);
            __half2_raw y1_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr0[1], __NV_E4M3);
            __half2_raw y2_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[0], __NV_E4M3);
            __half2_raw y3_raw = __nv_cvt_fp8x2_to_halfraw2(fp8_ptr1[1], __NV_E4M3);
            
            yh0 = *reinterpret_cast<half2*>(&y0_raw);
            yh1 = *reinterpret_cast<half2*>(&y1_raw);
            yh2 = *reinterpret_cast<half2*>(&y2_raw);
            yh3 = *reinterpret_cast<half2*>(&y3_raw);
            #else
            fp8x4_to_half2x2_lut(y_packed0, yh0, yh1);
            fp8x4_to_half2x2_lut(y_packed1, yh2, yh3);
            #endif
            
            half2 sum2 = __float2half2_rn(0.0f);
            sum2 = __hfma2(w01, yh0, sum2);
            sum2 = __hfma2(w23, yh1, sum2);
            sum2 = __hfma2(w45, yh2, sum2);
            sum2 = __hfma2(w67, yh3, sum2);
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    // -------------------------------------------------------------------------
    // DEQUANTIZE - Convert AWQ 4-bit to float
    // -------------------------------------------------------------------------
    template <int N>
    __device__ __forceinline__ void dequant(float* __restrict__ out) const {
        static_assert(N == 0, "Q_AWQ_G64 uses 1-part interface for K/128");
        
        const float scale = __half2float(__low2half(sm));
        const float neg_sz = __half2float(__high2half(sm));
        
        // LOP3-based extraction
        constexpr uint32_t LO_MASK = 0x000f000f;
        constexpr uint32_t EX = 0x64006400;
        constexpr uint32_t BIAS = 0x64006400;
        
        int raw01 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 0), LO_MASK, EX);
        int raw23 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 8), LO_MASK, EX);
        int raw45 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 4), LO_MASK, EX);
        int raw67 = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q_packed >> 12), LO_MASK, EX);
        
        half2 h01 = __hsub2(*reinterpret_cast<half2*>(&raw01), *reinterpret_cast<const half2*>(&BIAS));
        half2 h23 = __hsub2(*reinterpret_cast<half2*>(&raw23), *reinterpret_cast<const half2*>(&BIAS));
        half2 h45 = __hsub2(*reinterpret_cast<half2*>(&raw45), *reinterpret_cast<const half2*>(&BIAS));
        half2 h67 = __hsub2(*reinterpret_cast<half2*>(&raw67), *reinterpret_cast<const half2*>(&BIAS));
        
        // AWQ: w = scale * q + neg_sz
        float n0 = __half2float(__low2half(h01));
        float n1 = __half2float(__high2half(h01));
        float n2 = __half2float(__low2half(h23));
        float n3 = __half2float(__high2half(h23));
        float n4 = __half2float(__low2half(h45));
        float n5 = __half2float(__high2half(h45));
        float n6 = __half2float(__low2half(h67));
        float n7 = __half2float(__high2half(h67));
        
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        out4[0] = make_float4(
            __fmaf_rn(scale, n0, neg_sz),
            __fmaf_rn(scale, n1, neg_sz),
            __fmaf_rn(scale, n2, neg_sz),
            __fmaf_rn(scale, n3, neg_sz)
        );
        out4[1] = make_float4(
            __fmaf_rn(scale, n4, neg_sz),
            __fmaf_rn(scale, n5, neg_sz),
            __fmaf_rn(scale, n6, neg_sz),
            __fmaf_rn(scale, n7, neg_sz)
        );
    }
};

// =============================================================================
// TRAIT SPECIALIZATIONS
// =============================================================================
// Maps act_t → acc_t using: f32→f32, f16→f16, bf16→bf16, fp8→f16

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q_awq_g64, vdr, act_t> {
    using type = vec_dot_q_loader_q_awq_g64_inline<vdr, acc_for_act_t<act_t>>;
};

// =============================================================================
// GEMX DEQUANT TRAITS - Q_AWQ_G64 (4-bit AWQ with 2 scale/zero pairs per K/128)
// =============================================================================
// AWQ G64 K/128 layout (80 bytes):
//   qs[0..15]: 16 × int32 = 64 bytes (128 × 4-bit weights)
//   scales[2]: half at offset 64, 66
//   zeros[2]: half at offset 68, 70
//   _pad: 4 bytes padding (72-80)
//
// Dequant: w = scale * (q - zero) = scale * q + neg_sz
//   where neg_sz = -scale * zero
//
// GROUP MAPPING:
// - Group 0: k_iter 0-3 (K positions 0-63), uses scales[0]/zeros[0]
// - Group 1: k_iter 4-7 (K positions 64-127), uses scales[1]/zeros[1]
// =============================================================================

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q_awq_g64, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;  // AWQ has zero point (treated like neg_min)
    static constexpr int scales_per_ktile = 2;  // Two scales per K/128
    static constexpr int bits_per_element = 4;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // K/128 block byte size (80 = 64 qs + 4 scales + 4 zeros + 8 padding)
    static constexpr int K128_BYTES = 80;
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA m16n8k16
    // =========================================================================
    // AWQ G64 layout:
    // - qs[0..15]: 64 bytes of weights (16 threads × 8 nibbles each)
    // - scales[0] at offset 64, scales[1] at offset 66
    // - zeros[0] at offset 68, zeros[1] at offset 70
    //
    // Group selection: k_iter 0-3 → group 0, k_iter 4-7 → group 1
    // =========================================================================
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // Lane decomposition
        const int row = lane >> 2;
        const int k_group = lane & 3;
        
        // Base pointer for this row's K/128 block
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // AWQ G64 simple layout: qs[0..15] are contiguous 64 bytes
        const int qs_offset = k_iter * 8;
        
        // Load qs pair
        const int2 qs_pair = *reinterpret_cast<const int2*>(row_base + qs_offset);
        const int qs_lo = qs_pair.x;
        const int qs_hi = qs_pair.y;
        
        // Group selection: k_iter 0-3 → group 0, k_iter 4-7 → group 1
        const int group = k_iter >> 2;
        const half scale = *reinterpret_cast<const half*>(row_base + 64 + group * 2);
        const half zero = *reinterpret_cast<const half*>(row_base + 68 + group * 2);
        
        // Precompute neg_sz = -scale * zero
        const half neg_sz = __hmul(__hneg(scale), zero);
        
        // Shift: k_group → shift: 0→0, 1→8, 2→4, 3→12
        constexpr uint32_t LO_MASK = 0x000f000f;
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            const half2 scale2 = __half2half2(scale);
            const half2 neg_sz2 = __half2half2(neg_sz);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
            
            // w = scale * q + neg_sz
            frag[0] = __hfma2(scale2, raw0, neg_sz2);
            frag[1] = __hfma2(scale2, raw1, neg_sz2);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t BIAS = 0x43004300;
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(scale)));
            const __nv_bfloat162 neg_sz2 = __bfloat162bfloat162(__float2bfloat16(__half2float(neg_sz)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
            
            frag[0] = __hfma2(scale2, raw0, neg_sz2);
            frag[1] = __hfma2(scale2, raw1, neg_sz2);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            // FP8 uses FP16 MMA, keep in FP16
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            const half2 scale2 = __half2half2(scale);
            const half2 neg_sz2 = __half2half2(neg_sz);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
            
            half2 w0 = __hfma2(scale2, raw0, neg_sz2);
            half2 w1 = __hfma2(scale2, raw1, neg_sz2);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY UNROLLED)
    // =========================================================================
    // Optimizations:
    // 1. Lane math computed ONCE for all 4 slices
    // 2. All qs loads HOISTED upfront (ILP / memory latency hiding)
    // 3. Scale loaded ONCE per half (same for k_iter 0-3 or 4-7)
    // 4. half_idx as template parameter (compile-time branch elimination)
    //
    // AWQ G64 K/128 layout (72 bytes):
    //   qs[0..15] at 0-63, scales[2] at 64-67, zeros[2] at 68-71
    //
    // half_idx=0: k_iter 0-3 (offsets 0, 8, 16, 24), uses scales[0]/zeros[0]
    // half_idx=1: k_iter 4-7 (offsets 32, 40, 48, 56), uses scales[1]/zeros[1]
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
        
        // Shift: k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // =====================================================================
        // PHASE 2: HOIST ALL LOADS UPFRONT
        // =====================================================================
        // Load scale and zero for this half (compile-time constant group)
        const half scale = *reinterpret_cast<const half*>(row_base + 64 + half_idx * 2);
        const half zero = *reinterpret_cast<const half*>(row_base + 68 + half_idx * 2);
        const half neg_sz = __hmul(__hneg(scale), zero);
        
        // Load all 4 qs pairs upfront
        // half_idx=0: offsets 0, 8, 16, 24
        // half_idx=1: offsets 32, 40, 48, 56
        constexpr int qs0_off = (half_idx == 0) ? 0 : 32;
        constexpr int qs1_off = (half_idx == 0) ? 8 : 40;
        constexpr int qs2_off = (half_idx == 0) ? 16 : 48;
        constexpr int qs3_off = (half_idx == 0) ? 24 : 56;
        
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
            const half2 scale2 = __half2half2(scale);
            const half2 neg_sz2 = __half2half2(neg_sz);
            
            // k_iter 0 (or 4)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 1 (or 5)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 2 (or 6)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 3 (or 7)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t BIAS = 0x43004300;
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(scale)));
            const __nv_bfloat162 neg_sz2 = __bfloat162bfloat162(__float2bfloat16(__half2float(neg_sz)));
            
            // k_iter 0 (or 4)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_sz2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 1 (or 5)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_sz2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 2 (or 6)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_sz2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 3 (or 7)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&BIAS));
                __nv_bfloat162 w0 = __hfma2(scale2, raw0, neg_sz2);
                __nv_bfloat162 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
            
        } else {
            // FP8 path (uses FP16 dequant)
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            const half2 scale2 = __half2half2(scale);
            const half2 neg_sz2 = __half2half2(neg_sz);
            
            // k_iter 0 (or 4)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 1 (or 5)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 2 (or 6)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
            }
            // k_iter 3 (or 7)
            {
                int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.x >> shift), LO_MASK, EX);
                int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3.y >> shift), LO_MASK, EX);
                half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&BIAS));
                half2 w0 = __hfma2(scale2, raw0, neg_sz2);
                half2 w1 = __hfma2(scale2, raw1, neg_sz2);
                frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
            }
        }
    }
};
