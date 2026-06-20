#pragma once

// =============================================================================
// Q4_1 LOADER - K/128 FORMAT WITH EMBEDDED SCALES
// =============================================================================
//
// K/128 layout: 16 threads × 8 elements = 128 elements per block
// Each thread loads 8 × 4-bit weights from a single int (32 bits)
// Q4_1 has separate d and m scales (not derived like Q4_0)
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
struct vec_dot_q_loader_q4_1_inline {
    using acc2_type = acc2_for_act_t<acc_t>;
    
    int v;                   // 8 × 4-bit weights packed in 32 bits
    acc2_type dm;            // (d, m) in native format for acc_t
    
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;
    }
    
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q4_1* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q4_1 uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q4_1_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q4_1_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Q4_1 K/128 layout (80 bytes = 20 ints):
        // data[0-3]=qs0-3, data[4]=dm0, data[5-8]=qs4-7, data[9]=dm1,
        // data[10]=dm2, data[11-14]=qs8-11, data[15]=dm3, data[16-19]=qs12-15
        static constexpr int qs_idx[16] = {0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19};
        v = blk->data[qs_idx[lane]];
        
        // Scales: dm0 at data[4], dm1 at data[9], dm2 at data[10], dm3 at data[15]
        static constexpr int dm_idx[4] = {4, 9, 10, 15};
        const half2* dm_ptr = reinterpret_cast<const half2*>(&blk->data[dm_idx[lane >> 2]]);
        dm = convert_half2_to_acc2<acc2_type>(*dm_ptr);
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q4_1 uses 16-thread interface");
        
        const uint32_t q = v;
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH: Extract nibbles from LOP3-ready layout
            // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
            //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
            const float d = lo(dm);
            const float m = hi(dm);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // Extract nibbles from LOP3-ready positions
            const int n0 = q & 0xF;
            const int n1 = (q >> 16) & 0xF;
            const int n2 = (q >> 8) & 0xF;
            const int n3 = (q >> 24) & 0xF;
            const int n4 = (q >> 4) & 0xF;
            const int n5 = (q >> 20) & 0xF;
            const int n6 = (q >> 12) & 0xF;
            const int n7 = (q >> 28) & 0xF;
            
            float sum;
            // Elements 0-3
            {
                const float4 yv = y4[0];
                sum  = __fmaf_rn(__fmaf_rn(d, float(n0), m), yv.x, 0.0f);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n1), m), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n2), m), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n3), m), yv.w, sum);
            }
            // Elements 4-7
            {
                const float4 yv = y4[1];
                sum  = __fmaf_rn(__fmaf_rn(d, float(n4), m), yv.x, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n5), m), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n6), m), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(d, float(n7), m), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + SHIFT-BASED EXTRACTION (no PRMT needed)
            // LOP3-ready layout: v → (n0,n1), v>>8 → (n2,n3), v>>4 → (n4,n5), v>>12 → (n6,n7)
            const half2 d2 = __half2half2(lo_acc2(dm));
            const half2 m2 = __half2half2(hi_acc2(dm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // LOP3 magic constants for FP16
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x000f000f;
            
            // SHIFT-BASED EXTRACTION: Direct shifts produce half2-aligned pairs
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX);
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
            
            // LOP3 magic constants for BF16
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK = 0x000f000f;
            
            // SHIFT-BASED EXTRACTION: Direct shifts produce bf162-aligned pairs
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX_BF);
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
            
            // LOP3 weight dequantization with SHIFT-BASED EXTRACTION (no PRMT needed)
            // LOP3-ready layout: v → (n0,n1), v>>8 → (n2,n3), v>>4 → (n4,n5), v>>12 → (n6,n7)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x000f000f;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from v directly
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)q, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y0, sum2);
            }
            {
                // Pair 1: (n2, n3) from v >> 8
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 8), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y1, sum2);
            }
            {
                // Pair 2: (n4, n5) from v >> 4
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 4), LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2_v, sum2);
            }
            {
                // Pair 3: (n6, n7) from v >> 12
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(q >> 12), LO_MASK, EX);
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
        static_assert(N < 16, "Q4_1 uses 16-thread interface");
        
        const float d = to_f32(lo_acc2(dm));
        const float m = to_f32(hi_acc2(dm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = v;
        
        // Extract 8 nibbles from LOP3-ready packed layout
        // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
        //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
        const int n0 = q & 0xF;
        const int n1 = (q >> 16) & 0xF;
        const int n2 = (q >> 8) & 0xF;
        const int n3 = (q >> 24) & 0xF;
        const int n4 = (q >> 4) & 0xF;
        const int n5 = (q >> 20) & 0xF;
        const int n6 = (q >> 12) & 0xF;
        const int n7 = (q >> 28) & 0xF;
        
        // Output in sequential order: out[0..7] = dequant(n0, n1, n2, n3, n4, n5, n6, n7)
        out4[0] = make_float4(
            __fmaf_rn(d, float(n0), m),
            __fmaf_rn(d, float(n1), m),
            __fmaf_rn(d, float(n2), m),
            __fmaf_rn(d, float(n3), m)
        );
        out4[1] = make_float4(
            __fmaf_rn(d, float(n4), m),
            __fmaf_rn(d, float(n5), m),
            __fmaf_rn(d, float(n6), m),
            __fmaf_rn(d, float(n7), m)
        );
    }
};

template <int vdr, typename act_t>
struct vec_dot_loader_for<block_q4_1, vdr, act_t> {
    using type = vec_dot_q_loader_q4_1_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q4_1 (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q4_1, vdr, act_t> {
    using type = vec_dot_q_loader_q4_1_inline<vdr, acc_for_act_t<act_t>>;
};
// =============================================================================
// GEMX DEQUANT TRAITS - Q4_1 (4-bit asymmetric: value = d * q + m)
// =============================================================================
#include "gemx_dequant.cuh"

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q4_1, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;  // Explicit dm scale pairs
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q4_1>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 4;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q4_1 K/128 layout (80 bytes = 20 ints):
    //   data[0-3]=qs0-3, data[4]=dm0, data[5-8]=qs4-7, data[9]=dm1,
    //   data[10]=dm2, data[11-14]=qs8-11, data[15]=dm3, data[16-19]=qs12-15
    //
    // Scale groups: dm0 for qs0-3, dm1 for qs4-7, dm2 for qs8-11, dm3 for qs12-15
    // =========================================================================
    
    static constexpr int K128_BYTES = 80;

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 4-bit nibbles → n8k32 B-fragment (same LOP3 nibble order as Q4_0; one mask +
    // prmt 0x3120 → natural {0,1,2,3}). b_frag[0]/[1] come from two qs ints. The
    // fold applies d·C + m·Σx with the explicit affine {d, m} (value = d·q + m).
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        // data int index per qs field: qs0-3=data[0-3], qs4-7=data[5-8],
        // qs8-11=data[11-14], qs12-15=data[16-19].
        constexpr int QS_IDX[16] = {0, 1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 19};
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const int* data = reinterpret_cast<const int*>(warp_rows + row * K128_BYTES);
        const int sh = (q3 & 1) * 4;
        const int v0 = data[QS_IDX[sub * 4 + (q3 >> 1)]];
        const int v1 = data[QS_IDX[sub * 4 + 2 + (q3 >> 1)]];
        b_frag[0] = __byte_perm((v0 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        b_frag[1] = __byte_perm((v1 >> sh) & 0x0F0F0F0F, 0, 0x3120);
    }
    // Per-sub affine {d (low), m (high)} from dm0..dm3 at data[4,9,10,15].
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {16, 36, 40, 60};
        return *reinterpret_cast<const half2*>(row_block + DM_OFF[sub]);
    }

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
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS (one FragB) with compile-time shift
    // -------------------------------------------------------------------------
    
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
    // Q4_1 K/128 layout: 16 threads × 8 elements (4-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7 packed in one int (32 bits)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //
    // Q4_1 dequant: w = d * q + m where q is 0-15 (asymmetric)
    
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
        
        // Q4_1 K/128 byte layout (80 bytes):
        // Byte 0-15:  qs0-3 (4 ints, 16 bytes)
        // Byte 16-19: dm0 (half2, 4 bytes)
        // Byte 20-35: qs4-7 (4 ints, 16 bytes)
        // Byte 36-39: dm1 (half2, 4 bytes)
        // Byte 40-43: dm2 (half2, 4 bytes)
        // Byte 44-55: qs8-10 (3 ints, 12 bytes)
        // Byte 56-59: padding? check
        // Byte 60-63: dm3 (half2, 4 bytes)
        // Byte 64-79: qs12-15 (4 ints, 16 bytes)
        //
        // qs byte offsets for each thread:
        static constexpr int qs_byte_offset[16] = {
            0, 4, 8, 12,      // threads 0-3
            20, 24, 28, 32,   // threads 4-7
            44, 48, 52, 56,   // threads 8-11
            64, 68, 72, 76    // threads 12-15
        };
        // dm byte offsets: dm0=16, dm1=36, dm2=40, dm3=60
        static constexpr int dm_byte_offset[4] = {16, 36, 40, 60};
        
        // Thread indices for this k_iter
        const int thread_lo = k_iter << 1;      // k_iter * 2
        const int thread_hi = thread_lo + 1;
        const int scale_group = thread_lo >> 2; // which scale (0-3)
        
        // Load both qs values
        const int qs_lo = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_lo]);
        const int qs_hi = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_hi]);
        const half2 dm = *reinterpret_cast<const half2*>(row_base + dm_byte_offset[scale_group]);
        
        // =====================================================================
        // ELEMENT EXTRACTION
        // =====================================================================
        // MMA k_group determines which K positions:
        //   k_group=0: K = 0,1,8,9
        //   k_group=1: K = 2,3,10,11
        //   k_group=2: K = 4,5,12,13
        //   k_group=3: K = 6,7,14,15
        //
        // Q4_1 nibble layout in int (same as Q4_0):
        //   bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //   [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        //
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        constexpr int LO_MASK = 0x000f000f;
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q4_1: w = d * q + m (asymmetric quantization, no subtraction)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;  // Subtract 1024.0 to get raw q (0-15)
            const half2 d2 = __half2half2(__low2half(dm));
            const half2 m2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hfma2(d2, raw0, m2);  // d*q + m
            frag[1] = __hfma2(d2, raw1, m2);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;  // Subtract 128.0 to get raw q (0-15)
            const half d_h = __low2half(dm);
            const half m_h = __high2half(dm);
            const __nv_bfloat162 d2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_h)));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(__float2bfloat16(__half2float(m_h)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hfma2(d2, raw0, m2);
            frag[1] = __hfma2(d2, raw1, m2);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            const half2 d2 = __half2half2(__low2half(dm));
            const half2 m2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_lo >> shift), LO_MASK, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs_hi >> shift), LO_MASK, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            half2 w0 = __hfma2(d2, raw0, m2);
            half2 w1 = __hfma2(d2, raw1, m2);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (OPTIMIZED)
    // =========================================================================
    //
    // Q4_1 K/128 byte layout (80 bytes):
    //   Byte 0-15:  qs0-3 (4 ints, 16 bytes)
    //   Byte 16-19: dm0 (half2, 4 bytes)
    //   Byte 20-35: qs4-7 (4 ints, 16 bytes)
    //   Byte 36-39: dm1 (half2, 4 bytes)
    //   Byte 40-43: dm2 (half2, 4 bytes)
    //   Byte 44-59: qs8-11 (4 ints, 16 bytes)
    //   Byte 60-63: dm3 (half2, 4 bytes)
    //   Byte 64-79: qs12-15 (4 ints, 16 bytes)
    //
    // half_idx=0 (k_iter 0-3, threads 0-7):
    //   k_iter=0: qs @ 0,4   → dm @ 16   (int4 @ 0)
    //   k_iter=1: qs @ 8,12  → dm @ 16   (int2 @ 8)
    //   k_iter=2: qs @ 20,24 → dm @ 36   (2× int @ 20,24)
    //   k_iter=3: qs @ 28,32 → dm @ 36   (2× int @ 28,32)
    //
    // half_idx=1 (k_iter 4-7, threads 8-15):
    //   k_iter=4: qs @ 44,48 → dm @ 40   (2× int @ 44,48)
    //   k_iter=5: qs @ 52,56 → dm @ 40   (2× int @ 52,56)
    //   k_iter=6: qs @ 64,68 → dm @ 60   (int4 @ 64)
    //   k_iter=7: qs @ 72,76 → dm @ 60   (int2 @ 72)
    //
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
        half2 dm0, dm1;
        
        if constexpr (half_idx == 0) {
            // Load scales: dm0 @ 16, dm1 @ 36
            dm0 = *reinterpret_cast<const half2*>(row_base + 16);
            dm1 = *reinterpret_cast<const half2*>(row_base + 36);
            
            // k_iter 0,1: qs @ {0,4,8,12} - int4 @ 0 (16B aligned)
            {
                const int4 qs_vec = *reinterpret_cast<const int4*>(row_base + 0);
                qs0_lo = qs_vec.x;  // offset 0
                qs0_hi = qs_vec.y;  // offset 4
                qs1_lo = qs_vec.z;  // offset 8
                qs1_hi = qs_vec.w;  // offset 12
            }
            
            // k_iter 2,3: qs @ {20,24,28,32} - 4× int (20 not 8B aligned)
            qs2_lo = *reinterpret_cast<const int*>(row_base + 20);
            qs2_hi = *reinterpret_cast<const int*>(row_base + 24);
            qs3_lo = *reinterpret_cast<const int*>(row_base + 28);
            qs3_hi = *reinterpret_cast<const int*>(row_base + 32);
            
        } else {
            // Load scales: dm2 @ 40, dm3 @ 60
            dm0 = *reinterpret_cast<const half2*>(row_base + 40);
            dm1 = *reinterpret_cast<const half2*>(row_base + 60);
            
            // k_iter 4,5: qs @ {44,48,52,56} - 4× int (44 not 8B aligned)
            qs0_lo = *reinterpret_cast<const int*>(row_base + 44);
            qs0_hi = *reinterpret_cast<const int*>(row_base + 48);
            qs1_lo = *reinterpret_cast<const int*>(row_base + 52);
            qs1_hi = *reinterpret_cast<const int*>(row_base + 56);
            
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
        // Q4_1: w = d * q + m (asymmetric, uses hfma2)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // --- SCALE GROUP 0: dm0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const half2 d2 = __half2half2(__low2half(dm0));
                const half2 m2 = __half2half2(__high2half(dm0));
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: dm1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const half2 d2 = __half2half2(__low2half(dm1));
                const half2 m2 = __half2half2(__high2half(dm1));
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t EX = 0x43004300;
            constexpr uint32_t SUB = 0x43004300;
            
            // Convert scales once
            const float f_d0 = __half2float(__low2half(dm0));
            const float f_m0 = __half2float(__high2half(dm0));
            const float f_d1 = __half2float(__low2half(dm1));
            const float f_m1 = __half2float(__high2half(dm1));
            
            // --- SCALE GROUP 0: dm0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const __nv_bfloat162 d2 = __bfloat162bfloat162(__float2bfloat16(f_d0));
                const __nv_bfloat162 m2 = __bfloat162bfloat162(__float2bfloat16(f_m0));
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hfma2(d2, raw0, m2);
                    __nv_bfloat162 w1 = __hfma2(d2, raw1, m2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hfma2(d2, raw0, m2);
                    __nv_bfloat162 w1 = __hfma2(d2, raw1, m2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: dm1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const __nv_bfloat162 d2 = __bfloat162bfloat162(__float2bfloat16(f_d1));
                const __nv_bfloat162 m2 = __bfloat162bfloat162(__float2bfloat16(f_m1));
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hfma2(d2, raw0, m2);
                    __nv_bfloat162 w1 = __hfma2(d2, raw1, m2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
                    __nv_bfloat162 w0 = __hfma2(d2, raw0, m2);
                    __nv_bfloat162 w1 = __hfma2(d2, raw1, m2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr uint32_t EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // --- SCALE GROUP 0: dm0 for k_iter 0,1 → frag_b[0..3] ---
            {
                const half2 d2 = __half2half2(__low2half(dm0));
                const half2 m2 = __half2half2(__high2half(dm0));
                
                // k_iter 0
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs0_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 1
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs1_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
            // --- SCALE GROUP 1: dm1 for k_iter 2,3 → frag_b[4..7] ---
            {
                const half2 d2 = __half2half2(__low2half(dm1));
                const half2 m2 = __half2half2(__high2half(dm1));
                
                // k_iter 2
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs2_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter 3
                {
                    int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_lo >> shift), LO_MASK, EX);
                    int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>((uint32_t)(qs3_hi >> shift), LO_MASK, EX);
                    half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
                    half2 w0 = __hfma2(d2, raw0, m2);
                    half2 w1 = __hfma2(d2, raw1, m2);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
        }
    }
};

// Convenience aliases for Q4_1
using Q41_Dequant_FP16 = gemx_dequant_traits<block_c_q4_1, half, half>;
using Q41_Dequant_BF16 = gemx_dequant_traits<block_c_q4_1, __nv_bfloat16, __nv_bfloat16>;
using Q41_Dequant_FP8 = gemx_dequant_traits<block_c_q4_1, __nv_fp8_e4m3, half>;