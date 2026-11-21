// The Q4_K K-tile loader `vec_dot_q_loader_q4_K_inline` for the K/128 block
// with embedded scales, with its trait specializations.
//
// `vec_dot_q_loader_q4_K_inline` is the K-tile loader for Q4_K. The
// `vec_dot_loader_for` specializations, including the K/128 format alias,
// select it for each block type.

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
