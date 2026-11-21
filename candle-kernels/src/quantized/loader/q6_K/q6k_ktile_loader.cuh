// The Q6_K K-tile loader with its embedded-scale helpers and loader
// specializations.
//
// Holds the Q6_K helpers `q6k_elem_from_ql_dst`, `q6k_qh_dst_from_ql_dst`,
// `decode_q6k_scale_inline` and `load_q6k_external_scale`, the
// `block_c_q6_K_ktile` K-tile loader, `vec_dot_q_loader_q6_K_inline`, the
// `vec_dot_loader_for` specializations, and the `gemx_q6_K` scale extraction
// namespace.

// GEMX permutation helpers (unused in K-tile-major path)
__device__ __forceinline__ int q6k_elem_from_ql_dst(int ql_dst) {
    return (ql_dst >> 6) + (((ql_dst >> 1) & 31) << 2) + ((ql_dst & 1) << 7);
}

__device__ __forceinline__ int q6k_qh_dst_from_ql_dst(int ql_dst) {
    return (((ql_dst & 127) >> 1) << 2) + ((ql_dst >> 7) << 1) + (ql_dst & 1);
}

// Decodes scale from inline block structure (unused in K-tile-major path)
template <typename acc_t>
__device__ __forceinline__ acc_t decode_q6k_scale_inline(
    const half d,
    const int8_t* __restrict__ scales,
    int scale_idx
) {
    const float scale_f = __half2float(d) * float(scales[scale_idx]);
    if constexpr (std::is_same_v<acc_t, float>) {
        return scale_f;
    } else {
        return acc_t(scale_f);
    }
}

// Loads scale from GEMX-permuted external buffer (unused in K-tile-major path)
template <typename acc_t>
__device__ __forceinline__ acc_t load_q6k_external_scale(
    const half2* __restrict__ scales,
    int row,
    int kbx,
    int num_rows,
    int scale_idx
) {
    // External scales layout: 8 half2 pairs per superblock
    // pair_idx contains scales {2*pair_idx, 2*pair_idx+1}
    constexpr int SCALE_PAIRS_PER_SUPERBLOCK = 8;
    const int pair_idx = scale_idx / 2;
    
    const int scale_col = kbx * SCALE_PAIRS_PER_SUPERBLOCK + pair_idx;
    const int linear_idx = scale_col * num_rows + row;
    const int permuted_idx = gemx_permute_64(linear_idx);
    
    const half2 scale_pair = scales[permuted_idx];
    
    // Extract the correct scale from the pair
    const float scale_f = (scale_idx & 1) ? __high2float(scale_pair) : __low2float(scale_pair);
    
    if constexpr (std::is_same_v<acc_t, float>) {
        return scale_f;
    } else {
        return acc_t(scale_f);
    }
}

// =============================================================================
// K-TILE LOADER FOR Q6_K
// =============================================================================

// Local K-tile struct for loader storage (16 elements)
typedef struct __align__(4) {
    int ql_x;
    int ql_y;
    int qh;
} block_c_q6_K_ktile;

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q6_K_inline {
    // Option E: 8 elements per thread (single int of nibbles + 8 crumb bits)
    int ql;        // 8 nibbles (4 bits each)
    uint16_t qh8;  // 8 crumbs (2 bits each) = 16 bits (was uint32_t, saves 2B)
    half scale;
    
    // -------------------------------------------------------------------------
    // get_lane() returns 0-15 for indexing within a 16-thread group.
    // This does NOT mean half the threads are idle! The kernel processes
    // 8 K-blocks in parallel: threads 0-15 handle block 0, 16-31 handle block 1, etc.
    // All 128 threads are active (100% warp utilization).
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;  // Lane within 16-thread group (NOT warp lane!)
    }

    // 8-PART LOAD INTERFACE (8 elements per part) - Option E
    // 
    // COMPACT 112-BYTE LAYOUT: Contiguous arrays for perfect coalescing
    //   ql[16]    - bytes 0-63:   one int per thread
    //   qh[8]     - bytes 64-95:  one int per thread-pair (qh_lo | qh_hi << 16)
    //   scales[8] - bytes 96-111: one half per thread-pair
    //
    // vec_dot benefits: ql[lane] is perfectly contiguous across all 16 threads!
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q6_K* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q6_K uses 16-part interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q6_K_k128* __restrict__ blk = reinterpret_cast<const block_c_q6_K_k128*>(&x[block_idx]);

        const int lane = get_lane();
        const int pair = lane >> 1;       // 0-7: which thread-pair
        const int in_pair = lane & 1;     // 0 or 1: which thread within pair
        
        // ql: PERFECT COALESCING - each thread loads ql[lane]
        ql = blk->ql[lane];
        
        // qh: packed in qh[pair] as (qh_lo | qh_hi << 16)
        // Thread 0 of pair needs low 16 bits, thread 1 needs high 16 bits
        const uint32_t qh_packed = static_cast<uint32_t>(blk->qh[pair]);
        qh8 = (in_pair == 0) ? (qh_packed & 0xFFFF) : (qh_packed >> 16);
        
        // scale: one per thread-pair
        scale = blk->scales[pair];
    }
    
    // DOT PRODUCT - Option E: 8 elements per thread
    // Q6_K: 6-bit = 4-bit nibble + 2-bit crumb, symmetric quantization (q6 - 32)
    // 
    // LOP3-READY Memory layout (from repacker pack_nibbles_lop3_ready):
    //   ql bits: [3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
    //            [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
    //   qh8 uint16: c0c1c2c3c4c5c6c7 (each ci is 2 bits at position i*2)
    //
    // LOP3-READY Nibble extraction (with LO_MASK = 0x000f000f):
    //   ql & LO_MASK           → (n0, n1) pair
    //   (ql >> 8) & LO_MASK    → (n2, n3) pair
    //   (ql >> 4) & LO_MASK    → (n4, n5) pair
    //   (ql >> 12) & LO_MASK   → (n6, n7) pair
    //
    // We need to compute: sum_i(q6[i] * y[i]) where q6[i] = (nibble[i] | crumb[i]<<4) - 32
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 8, "Q6_K uses 8-part interface (Option E)");
        
        // LOP3-READY: Extract nibble pairs directly using shifts
        constexpr uint32_t LO_MASK = 0x000f000f;
        const uint32_t nib_01 = ql & LO_MASK;               // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;        // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;        // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;       // (n6, n7)
        
        // Extract crumbs from qh8 and build pairs matching nibble pairs
        // qh8 layout: c0(bits 0-1), c1(bits 2-3), c2(bits 4-5), ..., c7(bits 14-15)
        const uint32_t qh32 = static_cast<uint32_t>(qh8);
        
        // Build crumb pairs shifted left by 4 to OR with nibbles
        // crumb_01 = (c0 << 4) | (c1 << 20) for pairing with nib_01
        const uint32_t crumb_01 = ((qh32 & 0x3) << 4) | (((qh32 >> 2) & 0x3) << 20);
        const uint32_t crumb_23 = (((qh32 >> 4) & 0x3) << 4) | (((qh32 >> 6) & 0x3) << 20);
        const uint32_t crumb_45 = (((qh32 >> 8) & 0x3) << 4) | (((qh32 >> 10) & 0x3) << 20);
        const uint32_t crumb_67 = (((qh32 >> 12) & 0x3) << 4) | (((qh32 >> 14) & 0x3) << 20);
        
        // Combine nibbles and crumbs to form q6 pairs
        // Each q6 pair: (q6[i], q6[i+1]) where q6 = nibble | (crumb << 4)
        const uint32_t q6_01 = nib_01 | crumb_01;  // (q6[0], q6[1])
        const uint32_t q6_23 = nib_23 | crumb_23;  // (q6[2], q6[3])
        const uint32_t q6_45 = nib_45 | crumb_45;  // (q6[4], q6[5])
        const uint32_t q6_67 = nib_67 | crumb_67;  // (q6[6], q6[7])
        
        if constexpr (std::is_same_v<y_t, float>) {
            const float scale_f = __half2float(scale);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // Extract q6 values from pairs and compute inline
            // q6_XY has q6[X] in low 16 bits, q6[Y] in high 16 bits (only 6 bits used each)
            const int q0 = int(q6_01 & 0x3F) - 32;
            const int q1 = int((q6_01 >> 16) & 0x3F) - 32;
            const int q2 = int(q6_23 & 0x3F) - 32;
            const int q3 = int((q6_23 >> 16) & 0x3F) - 32;
            const int q4 = int(q6_45 & 0x3F) - 32;
            const int q5 = int((q6_45 >> 16) & 0x3F) - 32;
            const int q6 = int(q6_67 & 0x3F) - 32;
            const int q7 = int((q6_67 >> 16) & 0x3F) - 32;
            
            float sum = 0.0f;
            {
                const float4 yv = y4[0];
                sum  = __fmaf_rn(scale_f * float(q0), yv.x, sum);
                sum  = __fmaf_rn(scale_f * float(q1), yv.y, sum);
                sum  = __fmaf_rn(scale_f * float(q2), yv.z, sum);
                sum  = __fmaf_rn(scale_f * float(q3), yv.w, sum);
            }
            {
                const float4 yv = y4[1];
                sum  = __fmaf_rn(scale_f * float(q4), yv.x, sum);
                sum  = __fmaf_rn(scale_f * float(q5), yv.y, sum);
                sum  = __fmaf_rn(scale_f * float(q6), yv.z, sum);
                sum  = __fmaf_rn(scale_f * float(q7), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 OPTIMIZATION - No PRMT needed!
            // q6 pairs are already in LOP3-ready format: (q6[i], q6[i+1]) as half-words
            const half2 scale2 = __half2half2(scale);
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // LOP3 magic constants for FP16 signed conversion
            // EX = 0x64006400 (exp=10), BIAS = 0x64206420 = half2(1024+32, 1024+32)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS32 = 0x64206420;
            constexpr int LO_MASK_6BIT = 0x003f003f;  // Mask for 6-bit values
            
            // LOP3-READY: Direct conversion without PRMT
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_01, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_23, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_45, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_67, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 OPTIMIZATION - No PRMT needed!
            const __nv_bfloat16 scale_bf = __float2bfloat16(__half2float(scale));
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(scale_bf);
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // LOP3 magic constants for BF16 signed conversion
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS32_BF = 0x43204320;  // bf162(160, 160)
            constexpr int LO_MASK_6BIT = 0x003f003f;
            
            // LOP3-READY: Direct conversion without PRMT
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_01, LO_MASK_6BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS32_BF));
                sum2 = __hfma2(__hmul2(scale2, w), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_23, LO_MASK_6BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS32_BF));
                sum2 = __hfma2(__hmul2(scale2, w), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_45, LO_MASK_6BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS32_BF));
                sum2 = __hfma2(__hmul2(scale2, w), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_67, LO_MASK_6BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS32_BF));
                sum2 = __hfma2(__hmul2(scale2, w), y2[3], sum2);
            }
            
            return __bfloat162float(__low2bfloat16(sum2)) + __bfloat162float(__high2bfloat16(sum2));
            
        } else {
            // =================================================================
            // FP8 SPECIALIZED PATH - High performance via FP16 accumulation
            // =================================================================
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half2 scale2 = __half2half2(scale);
            
            // Load FP8 inputs as uint32_t (4 FP8 values per load)
            const uint32_t* y_u32 = reinterpret_cast<const uint32_t*>(y + get_lane() * 8);
            const uint32_t y_packed0 = y_u32[0];  // y[0..3] as FP8
            const uint32_t y_packed1 = y_u32[1];  // y[4..7] as FP8
            
            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            // Hardware FP8→FP16 conversion
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
            // Software conversion via LUT
            half2 y0, y1, y2_v, y3;
            fp8x4_to_half2x2_lut(y_packed0, y0, y1);
            fp8x4_to_half2x2_lut(y_packed1, y2_v, y3);
            #endif
            
            // LOP3-READY: Direct conversion without PRMT
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS32 = 0x64206420;
            constexpr int LO_MASK_6BIT = 0x003f003f;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_01, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y0, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_23, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y1, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_45, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y2_v, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q6_67, LO_MASK_6BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS32));
                sum2 = __hfma2(__hmul2(scale2, w), y3, sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    // DEQUANTIZE - Option E: Outputs 8 elements to out[lane*8 + 0..7] using float4 stores
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N < 8, "Q6_K uses 8-part interface (Option E)");
        
        const float scale_f = __half2float(scale);
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        
        // LOP3-READY: Extract nibble pairs using shifts
        constexpr uint32_t LO_MASK = 0x000f000f;
        const uint32_t nib_01 = ql & LO_MASK;               // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;        // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;        // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;       // (n6, n7)
        
        // Extract crumbs from qh8 and build pairs matching nibble pairs
        const uint32_t qh32 = static_cast<uint32_t>(qh8);
        const uint32_t crumb_01 = ((qh32 & 0x3) << 4) | (((qh32 >> 2) & 0x3) << 20);
        const uint32_t crumb_23 = (((qh32 >> 4) & 0x3) << 4) | (((qh32 >> 6) & 0x3) << 20);
        const uint32_t crumb_45 = (((qh32 >> 8) & 0x3) << 4) | (((qh32 >> 10) & 0x3) << 20);
        const uint32_t crumb_67 = (((qh32 >> 12) & 0x3) << 4) | (((qh32 >> 14) & 0x3) << 20);
        
        // Combine nibbles and crumbs to form q6 pairs
        const uint32_t q6_01 = nib_01 | crumb_01;
        const uint32_t q6_23 = nib_23 | crumb_23;
        const uint32_t q6_45 = nib_45 | crumb_45;
        const uint32_t q6_67 = nib_67 | crumb_67;
        
        // Extract individual values (6-bit each, in half-word positions)
        const float q0 = scale_f * float(int(q6_01 & 0x3F) - 32);
        const float q1 = scale_f * float(int((q6_01 >> 16) & 0x3F) - 32);
        const float q2 = scale_f * float(int(q6_23 & 0x3F) - 32);
        const float q3 = scale_f * float(int((q6_23 >> 16) & 0x3F) - 32);
        const float q4 = scale_f * float(int(q6_45 & 0x3F) - 32);
        const float q5 = scale_f * float(int((q6_45 >> 16) & 0x3F) - 32);
        const float q6 = scale_f * float(int(q6_67 & 0x3F) - 32);
        const float q7 = scale_f * float(int((q6_67 >> 16) & 0x3F) - 32);
        
        // Output in correct order: q0,q1,q2,q3, q4,q5,q6,q7
        out4[0] = make_float4(q0, q1, q2, q3);
        out4[1] = make_float4(q4, q5, q6, q7);
    }
};

// LOADER TRAIT SPECIALIZATIONS
template <typename act_t>
struct vec_dot_loader_for<block_q6_K, 1, act_t> {
    using type = vec_dot_q_loader_q6_K_inline<1, float>;
};

template <typename act_t>
struct vec_dot_loader_for<block_q6_K, 2, act_t> {
    using type = vec_dot_q_loader_q6_K_inline<2, float>;
};

// K/128 compact block type specializations
template <typename act_t>
struct vec_dot_loader_for<block_c_q6_K, 1, act_t> {
    using type = vec_dot_q_loader_q6_K_inline<1, float>;
};

template <typename act_t>
struct vec_dot_loader_for<block_c_q6_K, 2, act_t> {
    using type = vec_dot_q_loader_q6_K_inline<2, float>;
};

// =============================================================================
// SCALE EXTRACTION FOR REPACKING
// =============================================================================

namespace gemx_q6_K {

struct Q6K_Traits {
    static constexpr int BLOCK_ELEMENTS = 256;
    static constexpr int INPUT_BYTES = 210;
    static constexpr int OUTPUT_BYTES = 192;
    static constexpr int QL_BYTES = 128;
    static constexpr int QH_BYTES = 64;
    static constexpr bool NEEDS_PERMUTATION = true;
    static constexpr int THREADS_PER_BLOCK = 32;
};

// Extract single scale: d * scales[scale_idx] → half
template <typename ScaleT>
__device__ __forceinline__ void extract_scale(
    const block_q6_K* __restrict__ block,
    ScaleT* __restrict__ scales_out,
    int dst_scale_idx,
    int scale_idx
) {
    const float d = __half2float(block->d);
    const float s = d * float(block->scales[scale_idx]);
    scales_out[dst_scale_idx] = __float2half(s);
}

// Extract all scales: row-major [N, K/256] → column-major [K/16, N]
template <typename ScaleT>
__device__ inline void extract_scales_impl(
    const block_q6_K* __restrict__ x,
    ScaleT* __restrict__ scales_out,
    int nrows,
    int ncols
) {
    constexpr int SCALES_PER_SUPERBLOCK = 16;
    constexpr int ELEMENTS_PER_SUPERBLOCK = 256;
    constexpr int ELEMENTS_PER_SCALE = 16;
    const int superblocks_per_row = ncols / ELEMENTS_PER_SUPERBLOCK;
    const int scales_per_row = ncols / ELEMENTS_PER_SCALE;
    const int total_scales = nrows * scales_per_row;
    
    for (int src_scale_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         src_scale_idx < total_scales; 
         src_scale_idx += blockDim.x * gridDim.x) 
    {
        const int row = src_scale_idx / scales_per_row;
        const int scale_col = src_scale_idx % scales_per_row;
        const int superblock_col = scale_col / SCALES_PER_SUPERBLOCK;
        const int local_scale = scale_col % SCALES_PER_SUPERBLOCK;
        const int superblock_idx = row * superblocks_per_row + superblock_col;
        const int dst_scale_idx = scale_col * nrows + row;
        
        extract_scale<ScaleT>(&x[superblock_idx], scales_out, dst_scale_idx, local_scale);
    }
}

} // namespace gemx_q6_K
