// The Q5_1 K-tile loader `vec_dot_q_loader_q5_1_inline` for the K/128 block
// with embedded scales, with its trait specializations.
//
// `vec_dot_q_loader_q5_1_inline` is the Q5_1 K-tile loader. The
// `vec_dot_loader_for` specializations select it for each block type, and the
// alias names the `block_c_q5_1` type.

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q5_1_inline {
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
        const block_c_q5_1* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q5_1 uses 16-thread interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q5_1_k128* __restrict__ blk = 
            reinterpret_cast<const block_c_q5_1_k128*>(&x[block_idx]);
        
        const int lane = get_lane();
        
        // Lookup table for qs indices in data[] array:
        // threads 0-3 → data[0-3], threads 4-7 → data[6-9],
        // threads 8-11 → data[14-17], threads 12-15 → data[19-22]
        static constexpr int qs_idx[16] = {
            0, 1, 2, 3, 6, 7, 8, 9, 14, 15, 16, 17, 19, 20, 21, 22
        };
        
        // Lookup table for qh indices in data[] array:
        // group 0 (threads 0-3) → data[4], group 1 (threads 4-7) → data[10],
        // group 2 (threads 8-11) → data[13], group 3 (threads 12-15) → data[23]
        static constexpr int qh_idx[4] = {4, 10, 13, 23};
        
        // Lookup table for dm (half2) indices in data[] array:
        // group 0 → data[5], group 1 → data[11], group 2 → data[12], group 3 → data[18]
        static constexpr int dm_idx[4] = {5, 11, 12, 18};
        
        // Load qs (8 x 4-bit low nibbles)
        qs = blk->data[qs_idx[lane]];
        
        // Load qh - packed as 4 uint8_t per int, extract this thread's byte
        const int qh_group = lane >> 2;
        const int qh_in_group = lane & 3;
        const int qh_packed = blk->data[qh_idx[qh_group]];
        qh = (qh_packed >> (qh_in_group * 8)) & 0xFF;
        
        // Load dm (half2) - 4 threads share each scale pair
        const int scale_group = lane >> 2;
        const half2 dm_raw = *reinterpret_cast<const half2*>(&blk->data[dm_idx[scale_group]]);
        dm = convert_half2_to_acc2<acc2_type>(dm_raw);
    }
    
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q5_1 uses 16-thread interface");
        
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        if constexpr (std::is_same_v<y_t, float>) {
            // FLOAT PATH: Extract from LOP3-ready layout + high bits
            // LOP3-ready layout: bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6
            //                    bits[19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
            const float d = lo(dm);
            const float m = hi(dm);
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            // Extract low nibbles from LOP3-ready positions
            const int nl0 = q & 0xF;           // bits[3:0]
            const int nl1 = (q >> 16) & 0xF;   // bits[19:16]
            const int nl2 = (q >> 8) & 0xF;    // bits[11:8]
            const int nl3 = (q >> 24) & 0xF;   // bits[27:24]
            const int nl4 = (q >> 4) & 0xF;    // bits[7:4]
            const int nl5 = (q >> 20) & 0xF;   // bits[23:20]
            const int nl6 = (q >> 12) & 0xF;   // bits[15:12]
            const int nl7 = (q >> 28) & 0xF;   // bits[31:28]
            
            float sum = 0.0f;
            // Elements 0-3 (combine low nibbles with high bits)
            {
                const float4 yv = y4[0];
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl0, (h >> 0) & 1)), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl1, (h >> 1) & 1)), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl2, (h >> 2) & 1)), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl3, (h >> 3) & 1)), m), yv.w, sum);
            }
            // Elements 4-7
            {
                const float4 yv = y4[1];
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl4, (h >> 4) & 1)), m), yv.x, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl5, (h >> 5) & 1)), m), yv.y, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl6, (h >> 6) & 1)), m), yv.z, sum);
                sum = __fmaf_rn(__fmaf_rn(d, float(get_q5(nl7, (h >> 7) & 1)), m), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + SHIFT-BASED EXTRACTION (no PRMT needed)
            // LOP3-ready layout: shift-based extraction gives half2-aligned pairs
            const half2 d2 = __half2half2(lo_acc2(dm));
            const half2 m2 = __half2half2(hi_acc2(dm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // Pre-compute high bit pairs aligned for half2 processing:
            // y2[0] = (y[0], y[1]) needs q5 values for elements 0,1: hb_01 = (h0<<4) | (h1<<20)
            // y2[1] = (y[2], y[3]) needs q5 values for elements 2,3: hb_23 = (h2<<4) | (h3<<20)
            // y2[2] = (y[4], y[5]) needs q5 values for elements 4,5: hb_45 = (h4<<4) | (h5<<20)
            // y2[3] = (y[6], y[7]) needs q5 values for elements 6,7: hb_67 = (h6<<4) | (h7<<20)
            const uint32_t hb_01 = ((h & 0x01) << 4) | (((h >> 1) & 0x01) << 20);
            const uint32_t hb_23 = (((h >> 2) & 0x01) << 4) | (((h >> 3) & 0x01) << 20);
            const uint32_t hb_45 = (((h >> 4) & 0x01) << 4) | (((h >> 5) & 0x01) << 20);
            const uint32_t hb_67 = (((h >> 6) & 0x01) << 4) | (((h >> 7) & 0x01) << 20);
            
            // LOP3 magic constants for FP16 (5-bit values: 0-31)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x001f001f;  // 5-bit mask
            
            // SHIFT-BASED EXTRACTION from LOP3-ready layout:
            // q >> 0  → (n0, n1) + hb_01 → y2[0]
            // q >> 8  → (n2, n3) + hb_23 → y2[1]
            // q >> 4  → (n4, n5) + hb_45 → y2[2]
            // q >> 12 → (n6, n7) + hb_67 → y2[3]
            half2 sum2 = __float2half2_rn(0.0f);
            {
                // Pair 0: (n0, n1) from q directly + high bits
                const uint32_t q5_pair = ((q >> 0) & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                // Pair 1: (n2, n3) from q >> 8 + high bits
                const uint32_t q5_pair = ((q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                // Pair 2: (n4, n5) from q >> 4 + high bits
                const uint32_t q5_pair = ((q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                // Pair 3: (n6, n7) from q >> 12 + high bits
                const uint32_t q5_pair = ((q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 + SHIFT-BASED EXTRACTION (no PRMT needed)
            const __nv_bfloat162 d2 = __bfloat162bfloat162(lo_acc2(dm));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(hi_acc2(dm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // Pre-compute high bit pairs aligned for bf162 processing
            const uint32_t hb_01 = ((h & 0x01) << 4) | (((h >> 1) & 0x01) << 20);
            const uint32_t hb_23 = (((h >> 2) & 0x01) << 4) | (((h >> 3) & 0x01) << 20);
            const uint32_t hb_45 = (((h >> 4) & 0x01) << 4) | (((h >> 5) & 0x01) << 20);
            const uint32_t hb_67 = (((h >> 6) & 0x01) << 4) | (((h >> 7) & 0x01) << 20);
            
            // LOP3 magic constants for BF16 (5-bit values: 0-31)
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;
            constexpr int LO_MASK = 0x001f001f;  // 5-bit mask
            
            // SHIFT-BASED EXTRACTION from LOP3-ready layout
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                const uint32_t q5_pair = ((q >> 0) & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[0], sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[1], sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2[2], sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX_BF);
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
            
            // SHIFT-BASED EXTRACTION from LOP3-ready layout with high bits
            // Pre-compute high bit pairs aligned for half2 processing
            const uint32_t hb_01 = ((h & 0x01) << 4) | (((h >> 1) & 0x01) << 20);
            const uint32_t hb_23 = (((h >> 2) & 0x01) << 4) | (((h >> 3) & 0x01) << 20);
            const uint32_t hb_45 = (((h >> 4) & 0x01) << 4) | (((h >> 5) & 0x01) << 20);
            const uint32_t hb_67 = (((h >> 6) & 0x01) << 4) | (((h >> 7) & 0x01) << 20);
            
            // LOP3 weight dequantization (5-bit values: 0-31)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK = 0x001f001f;
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                const uint32_t q5_pair = ((q >> 0) & 0x000f000f) | hb_01;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y0, sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 8) & 0x000f000f) | hb_23;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y1, sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 4) & 0x000f000f) | hb_45;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(d2, w, m2), y2_v, sum2);
            }
            {
                const uint32_t q5_pair = ((q >> 12) & 0x000f000f) | hb_67;
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_pair, LO_MASK, EX);
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
        static_assert(N < 16, "Q5_1 uses 16-thread interface");
        
        const float d = to_f32(lo_acc2(dm));
        const float m = to_f32(hi_acc2(dm));
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        const uint32_t q = qs;
        const uint32_t h = qh;
        
        // Extract low nibbles from LOP3-ready positions
        // Layout: bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6
        //         bits[19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        const int nl0 = q & 0xF;           // bits[3:0]
        const int nl1 = (q >> 16) & 0xF;   // bits[19:16]
        const int nl2 = (q >> 8) & 0xF;    // bits[11:8]
        const int nl3 = (q >> 24) & 0xF;   // bits[27:24]
        const int nl4 = (q >> 4) & 0xF;    // bits[7:4]
        const int nl5 = (q >> 20) & 0xF;   // bits[23:20]
        const int nl6 = (q >> 12) & 0xF;   // bits[15:12]
        const int nl7 = (q >> 28) & 0xF;   // bits[31:28]
        
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
struct vec_dot_loader_for<block_q5_1, vdr, act_t> {
    using type = vec_dot_q_loader_q5_1_inline<vdr, acc_for_act_t<act_t>>;
};

// Alias for block_c_q5_1 (K/128 format typedef)
template <int vdr, typename act_t>
struct vec_dot_loader_for<block_c_q5_1, vdr, act_t> {
    using type = vec_dot_q_loader_q5_1_inline<vdr, acc_for_act_t<act_t>>;
};
