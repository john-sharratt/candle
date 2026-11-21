// The GEMX dequant traits for Q5_0 and its convenience aliases.
//
// `gemx_dequant_traits` for Q5_0, where a value is `d * (q5 - 16)`.
// `Q50_Dequant_FP16`, `Q50_Dequant_BF16` and `Q50_Dequant_FP8` are the
// convenience aliases for the three instantiations.

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_0, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;  // m = -16*d (computed from d)
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q5_0>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 5;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q5_0 K/128 layout (112 bytes = 28 ints):
    //   data[0-3]=qs0-3, data[4]=qh0123, data[5]=d0|_spad, data[6]=_spad,
    //   data[7-10]=qs4-7, data[11]=qh4567, data[12]=d1|d2, data[13]=qh891011,
    //   data[14-17]=qs8-11, data[18]=_spad, data[19]=d3|_spad,
    //   data[20-23]=qs12-15, data[24]=qh12131415
    //
    // Scale groups: d0 for qs0-3, d1 for qs4-7, d2 for qs8-11, d3 for qs12-15
    // qh contains high bits: 4 threads × 8 bits = 4 bytes per qh word
    // =========================================================================
    
    static constexpr int K128_BYTES = 112;

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 5-bit = 4-bit nibble (qs) + 1 high bit (qh). Nibbles unpack like Q4_0 (mask +
    // prmt 0x3120 → natural {0,1,2,3}); the high bit per element is spread into bit 4
    // of each output byte via (hb·0x00204081 & 0x01010101)<<4. The fold applies
    // d·C + m·Σx with m = -16·d (Q5_0 centers the 5-bit value by -16).
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_OFF[16] = {0, 4, 8, 12, 28, 32, 36, 40, 56, 60, 64, 68, 80, 84, 88, 92};
        constexpr int QH_BASE[4] = {16, 44, 52, 96};
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const uint8_t* rb = warp_rows + row * K128_BYTES;
        const int sh = (q3 & 1) * 4;
        const int m0 = sub * 4 + (q3 >> 1);
        const int m1 = m0 + 2;
        const int v0 = *reinterpret_cast<const int*>(rb + QS_OFF[m0]);
        const int v1 = *reinterpret_cast<const int*>(rb + QS_OFF[m1]);
        const uint32_t nib0 = __byte_perm((v0 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t nib1 = __byte_perm((v1 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t qh0 = *(rb + QH_BASE[m0 >> 2] + (m0 & 3));
        const uint32_t qh1 = *(rb + QH_BASE[m1 >> 2] + (m1 & 3));
        const uint32_t hb0 = (qh0 >> sh) & 0xF;
        const uint32_t hb1 = (qh1 >> sh) & 0xF;
        b_frag[0] = nib0 | (((hb0 * 0x00204081u) & 0x01010101u) << 4);
        b_frag[1] = nib1 | (((hb1 * 0x00204081u) & 0x01010101u) << 4);
    }
    // Per-sub {scale d (low), neg_min = -16·d (high)}.
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int SCALE_OFF[4] = {20, 48, 50, 76};
        const half d = *reinterpret_cast<const half*>(row_block + SCALE_OFF[sub]);
        return __halves2half2(d, __hmul(d, __float2half(-16.0f)));
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
    
    // Combine low nibble with high bit to form 5-bit value
    __device__ __forceinline__ static int make_q5(int low4, int high1) {
        return low4 | (high1 << 4);
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 Q5 ELEMENTS (one FragB) with compile-time shift
    // -------------------------------------------------------------------------
    // Need to combine low nibbles from qs with high bits from qh
    // LOP3-ready layout: v >> 0 → (n0,n1), v >> 8 → (n2,n3), v >> 4 → (n4,n5), v >> 12 → (n6,n7)
    // For NIBBLE_HALF=0: extract (n0,n1,n2,n3) with h bits 0,1,2,3
    // For NIBBLE_HALF=1: extract (n4,n5,n6,n7) with h bits 4,5,6,7
    
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(int q, uint8_t h, FragB_t& frag) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int SHIFT0 = (NIBBLE_HALF == 0) ? 0 : 4;
        constexpr int SHIFT1 = (NIBBLE_HALF == 0) ? 8 : 12;
        constexpr int H_BASE = (NIBBLE_HALF == 0) ? 0 : 4;
        
        // Build high-bit pairs for half2 alignment
        // For pair (a,b): hb = (h_a << 4) | (h_b << 20)
        const uint32_t hb_01 = (((h >> (H_BASE + 0)) & 1) << 4) | (((h >> (H_BASE + 1)) & 1) << 20);
        const uint32_t hb_23 = (((h >> (H_BASE + 2)) & 1) << 4) | (((h >> (H_BASE + 3)) & 1) << 20);
        
        // Extract low nibbles and OR with high bits
        const uint32_t q5_01 = (((uint32_t)q >> SHIFT0) & 0x000f000f) | hb_01;
        const uint32_t q5_23 = (((uint32_t)q >> SHIFT1) & 0x000f000f) | hb_23;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_FP16);
            
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
    // EXTRACT 8 Q5 ELEMENTS (two FragB) for K=32 MMA (FP8)
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(int q, uint8_t h, FragB_t& frag0, FragB_t& frag1) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        // Build all high-bit pairs
        const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
        const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
        const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
        const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
        
        // Extract low nibbles and OR with high bits
        const uint32_t q5_01 = ((uint32_t)q & 0x000f000f) | hb_01;
        const uint32_t q5_23 = (((uint32_t)q >> 8) & 0x000f000f) | hb_23;
        const uint32_t q5_45 = (((uint32_t)q >> 4) & 0x000f000f) | hb_45;
        const uint32_t q5_67 = (((uint32_t)q >> 12) & 0x000f000f) | hb_67;
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX);
            
            frag0[0] = __hsub2(*reinterpret_cast<half2*>(&w01), *reinterpret_cast<const half2*>(&SUB));
            frag0[1] = __hsub2(*reinterpret_cast<half2*>(&w23), *reinterpret_cast<const half2*>(&SUB));
            frag1[0] = __hsub2(*reinterpret_cast<half2*>(&w45), *reinterpret_cast<const half2*>(&SUB));
            frag1[1] = __hsub2(*reinterpret_cast<half2*>(&w67), *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_BF);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX_BF);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX_BF);
            
            frag0[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag0[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w45), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag1[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w67), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_FP16);
            int w45 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX_FP16);
            int w67 = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX_FP16);
            
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
    // Q5_0 K/128 layout: 16 threads × 8 elements (5-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7: low 4 bits in qs (int), high bits in qh (uint8_t)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //
    // Q5_0 dequant: w = d * (q5 - 16) where q5 is 0-31 (symmetric)
    
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
        
        // Q5_0 K/128 byte layout (112 bytes):
        // Based on struct: qs0-3, qh0123, d0, _spad0, qs4-7, qh4567, d1, d2, qh891011, qs8-11, _spad1, d3, qs12-15, qh12131415
        //
        // qs byte offsets for each thread:
        static constexpr int qs_byte_offset[16] = {
            0, 4, 8, 12,      // threads 0-3 (qs0-3)
            28, 32, 36, 40,   // threads 4-7 (qs4-7)
            56, 60, 64, 68,   // threads 8-11 (qs8-11)
            80, 84, 88, 92    // threads 12-15 (qs12-15)
        };
        // qh byte offsets: qh0123=16, qh4567=44, qh891011=52, qh12131415=96
        static constexpr int qh_byte_offset[4] = {16, 44, 52, 96};
        // d byte offsets: d0=20, d1=48, d2=50, d3=76
        static constexpr int d_byte_offset[4] = {20, 48, 50, 76};
        
        // Thread indices for this k_iter
        const int thread_lo = k_iter << 1;      // k_iter * 2
        const int thread_hi = thread_lo + 1;
        const int scale_group = thread_lo >> 2; // which scale (0-3)
        const int thread_in_group_lo = thread_lo & 3;
        const int thread_in_group_hi = thread_hi & 3;
        
        // Load both qs and qh values
        const int qs_lo = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_lo]);
        const int qs_hi = *reinterpret_cast<const int*>(row_base + qs_byte_offset[thread_hi]);
        const int qh_packed = *reinterpret_cast<const int*>(row_base + qh_byte_offset[scale_group]);
        const uint8_t h_lo = (qh_packed >> (thread_in_group_lo * 8)) & 0xFF;
        const uint8_t h_hi = (qh_packed >> (thread_in_group_hi * 8)) & 0xFF;
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
        // Q5_0 nibble layout in int (same as Q4):
        //   bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //   [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        //
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // k_group → h_base: 0→0, 1→2, 2→4, 3→6
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;  // Each k_group needs 2 adjacent h bits
        
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        // Build 5-bit values: low 4 bits from qs, high bit from qh
        // For qs_lo: extract 2 nibbles (elements n, n+1 from k_group's position)
        // h_lo bits layout: bit i corresponds to element i of the 8 elements
        // For k_group 0: need bits 0,1 for elements 0,1
        // For k_group 1: need bits 2,3 for elements 2,3
        // etc.
        const uint32_t lo_nibbles = ((uint32_t)qs_lo >> shift) & 0x000f000f;
        const uint32_t hi_nibbles = ((uint32_t)qs_hi >> shift) & 0x000f000f;
        
        // High bits: bit h_base goes to element 0, bit h_base+1 goes to element 1
        const uint32_t hb_lo = (((h_lo >> h_base) & 1) << 4) | (((h_lo >> (h_base + 1)) & 1) << 20);
        const uint32_t hb_hi = (((h_hi >> h_base) & 1) << 4) | (((h_hi >> (h_base + 1)) & 1) << 20);
        
        const uint32_t q5_lo = lo_nibbles | hb_lo;
        const uint32_t q5_hi = hi_nibbles | hb_hi;
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q5_0: w = d * (q5 - 16) (symmetric quantization)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64106410;  // Subtract 1040.0 (1024+16) in fp16
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43104310;  // Subtract 144.0 (128+16) in bf16
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64106410;
            const half2 scale2 = __half2half2(d);
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
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
    // Processes 4 consecutive k_iters with vector loads and hoisted computations.
    // Key optimizations:
    // - Single shift/h_base computation (constant across all k_iters)
    // - Vector loads where alignment permits: int4 (16B), int2 (8B), int (4B)
    // - Shared qh/d loads for k_iter pairs (same scale_group)
    // - Q5_0 symmetric dequant: w = d * (q5 - 16) uses hmul2 (not hfma2)
    // =========================================================================
    
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        // =====================================================================
        // PHASE 1: LANE-CONSTANT COMPUTATIONS (hoisted from loop)
        // =====================================================================
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Shift pattern: k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;
        
        // LOP3 constants (compile-time)
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int EX_FP16 = 0x64006400;
        constexpr uint32_t SUB_FP16 = 0x64106410;  // 1024+16 for symmetric
        constexpr int EX_BF16 = 0x43004300;
        constexpr uint32_t SUB_BF16 = 0x43104310;  // 128+16 for symmetric
        
        if constexpr (half_idx == 0) {
            // =================================================================
            // HALF 0: k_iter 0,1,2,3 → threads 0-7
            // =================================================================
            // qs: threads 0-3 @ offsets 0,4,8,12 (16B aligned → int4)
            //     threads 4-7 @ offsets 28,32,36,40 (4B aligned → 4× int)
            // qh: scale_group 0 @ offset 16, scale_group 1 @ offset 44
            // d:  scale_group 0 @ offset 20, scale_group 1 @ offset 48
            
            // =====================================================================
            // PHASE 2: HOIST ALL LOADS UPFRONT (memory latency hiding)
            // =====================================================================
            const int4 qs_01 = *reinterpret_cast<const int4*>(row_base + 0);   // qs[0-3]
            const int qs_4 = *reinterpret_cast<const int*>(row_base + 28);
            const int qs_5 = *reinterpret_cast<const int*>(row_base + 32);
            const int qs_6 = *reinterpret_cast<const int*>(row_base + 36);
            const int qs_7 = *reinterpret_cast<const int*>(row_base + 40);
            
            const int qh_01 = *reinterpret_cast<const int*>(row_base + 16);
            const int qh_23 = *reinterpret_cast<const int*>(row_base + 44);
            const half d_01 = *reinterpret_cast<const half*>(row_base + 20);
            const half d_23 = *reinterpret_cast<const half*>(row_base + 48);
            
            // Extract h bytes
            const uint8_t h0 = qh_01 & 0xFF, h1 = (qh_01 >> 8) & 0xFF;
            const uint8_t h2 = (qh_01 >> 16) & 0xFF, h3 = (qh_01 >> 24) & 0xFF;
            const uint8_t h4 = qh_23 & 0xFF, h5 = (qh_23 >> 8) & 0xFF;
            const uint8_t h6 = (qh_23 >> 16) & 0xFF, h7 = (qh_23 >> 24) & 0xFF;
            
            // =====================================================================
            // PHASE 3: TYPE-SPECIFIC DEQUANTIZATION
            // =====================================================================
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 scale2_01 = __half2half2(d_01);
                const half2 scale2_23 = __half2half2(d_23);
                
                // k_iter=0: threads 0,1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1: threads 2,3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2: threads 4,5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3: threads 6,7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 scale2_01 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_01)));
                const __nv_bfloat162 scale2_23 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_23)));
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 scale2_01 = __half2half2(d_01);
                const half2 scale2_23 = __half2half2(d_23);
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_01, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_5 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_6 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_7 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_23, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
            
        } else {
            // =================================================================
            // HALF 1: k_iter 4,5,6,7 → threads 8-15
            // =================================================================
            // qs: threads 8-11 @ offsets 56,60,64,68 (8B aligned → 2× int2)
            //     threads 12-15 @ offsets 80,84,88,92 (16B aligned → int4)
            // qh: scale_group 2 @ offset 52, scale_group 3 @ offset 96
            // d:  scale_group 2 @ offset 50, scale_group 3 @ offset 76
            
            const int2 qs_4a = *reinterpret_cast<const int2*>(row_base + 56);  // qs[8-9]
            const int2 qs_4b = *reinterpret_cast<const int2*>(row_base + 64);  // qs[10-11]
            const int4 qs_67 = *reinterpret_cast<const int4*>(row_base + 80);  // qs[12-15]
            
            const int qh_45 = *reinterpret_cast<const int*>(row_base + 52);
            const int qh_67 = *reinterpret_cast<const int*>(row_base + 96);
            const half d_45 = *reinterpret_cast<const half*>(row_base + 50);
            const half d_67 = *reinterpret_cast<const half*>(row_base + 76);
            
            const uint8_t h8 = qh_45 & 0xFF, h9 = (qh_45 >> 8) & 0xFF;
            const uint8_t h10 = (qh_45 >> 16) & 0xFF, h11 = (qh_45 >> 24) & 0xFF;
            const uint8_t h12 = qh_67 & 0xFF, h13 = (qh_67 >> 8) & 0xFF;
            const uint8_t h14 = (qh_67 >> 16) & 0xFF, h15 = (qh_67 >> 24) & 0xFF;
            
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 scale2_45 = __half2half2(d_45);
                const half2 scale2_67 = __half2half2(d_67);
                
                // k_iter=4: threads 8,9
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5: threads 10,11
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6: threads 12,13
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7: threads 14,15
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 scale2_45 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_45)));
                const __nv_bfloat162 scale2_67 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_67)));
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    __nv_bfloat162 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 scale2_45 = __half2half2(d_45);
                const half2 scale2_67 = __half2half2(d_67);
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_45, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&w1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_67.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_67.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    half2 w1 = __hmul2(scale2_67, __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16)));
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&w0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&w1);
                }
            }
        }
    }
};

// Convenience aliases for Q5_0
using Q50_Dequant_FP16 = gemx_dequant_traits<block_c_q5_0, half, half>;
using Q50_Dequant_BF16 = gemx_dequant_traits<block_c_q5_0, __nv_bfloat16, __nv_bfloat16>;
using Q50_Dequant_FP8 = gemx_dequant_traits<block_c_q5_0, __nv_fp8_e4m3, half>;