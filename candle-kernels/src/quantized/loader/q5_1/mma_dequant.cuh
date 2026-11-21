// The GEMX dequant traits for Q5_1 and its convenience aliases.
//
// `gemx_dequant_traits` for Q5_1, where a value is `d * q5 + m`.
// `Q51_Dequant_FP16`, `Q51_Dequant_BF16` and `Q51_Dequant_FP8` are the
// convenience aliases for the three instantiations.

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_1, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = true;  // Explicit dm scale pairs
    static constexpr int scales_per_ktile = gemx_tile_traits<block_c_q5_1>::scales_per_ktile;  // 4
    static constexpr int bits_per_element = 5;
    
    // Fragment types
    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    using FragS = typename Frags::FragS;
    using vec2_t = typename Frags::vec2_t;
    
    using constants = dequant_constants<compute_t>;
    
    // =========================================================================
    // Q5_1 K/128 layout (112 bytes = 28 ints):
    //   data[0-3]=qs0-3, data[4]=qh0123, data[5]=dm0, data[6-9]=qs4-7,
    //   data[10]=qh4567, data[11]=dm1, data[12]=dm2, data[13]=qh891011,
    //   data[14-17]=qs8-11, data[18]=dm3, data[19-22]=qs12-15, data[23]=qh12131415
    //
    // Scale groups: dm0 for qs0-3, dm1 for qs4-7, dm2 for qs8-11, dm3 for qs12-15
    // =========================================================================
    
    static constexpr int K128_BYTES = 112;

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 5-bit = 4-bit nibble (qs) + 1 high bit (qh). Nibbles unpack like Q4_1 (mask +
    // prmt 0x3120); the high bit per element is spread into bit 4 of each output byte
    // via (hb·0x00204081 & 0x01010101)<<4. The fold applies d·C + m·Σx with the
    // explicit affine {d, m} (Q5_1 value = d·q + m, q in 0..31).
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_OFF[16] = {0, 4, 8, 12, 24, 28, 32, 36, 56, 60, 64, 68, 76, 80, 84, 88};
        constexpr int QH_BASE[4] = {16, 40, 52, 92};
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
    // Per-sub affine {d (low), m (high)} from dm0..dm3 at data[5,11,12,18].
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {20, 44, 48, 72};
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
    // EXTRACT 4 Q5 ELEMENTS (one FragB) - reuse from Q5_0
    // -------------------------------------------------------------------------
    
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(int q, uint8_t h, FragB_t& frag) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int SHIFT0 = (NIBBLE_HALF == 0) ? 0 : 4;
        constexpr int SHIFT1 = (NIBBLE_HALF == 0) ? 8 : 12;
        constexpr int H_BASE = (NIBBLE_HALF == 0) ? 0 : 4;
        
        const uint32_t hb_01 = (((h >> (H_BASE + 0)) & 1) << 4) | (((h >> (H_BASE + 1)) & 1) << 20);
        const uint32_t hb_23 = (((h >> (H_BASE + 2)) & 1) << 4) | (((h >> (H_BASE + 3)) & 1) << 20);
        
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
    // EXTRACT 8 Q5 ELEMENTS (two FragB) for K=32 MMA
    // -------------------------------------------------------------------------
    
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(int q, uint8_t h, FragB_t& frag0, FragB_t& frag1) {
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        const uint32_t hb_01 = (((h >> 0) & 1) << 4) | (((h >> 1) & 1) << 20);
        const uint32_t hb_23 = (((h >> 2) & 1) << 4) | (((h >> 3) & 1) << 20);
        const uint32_t hb_45 = (((h >> 4) & 1) << 4) | (((h >> 5) & 1) << 20);
        const uint32_t hb_67 = (((h >> 6) & 1) << 4) | (((h >> 7) & 1) << 20);
        
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
    // Q5_1 K/128 layout: 16 threads × 8 elements (5-bit each) = 128 elements
    // Thread t has elements t*8 to t*8+7: low 4 bits in qs (int), high bits in qh (uint8_t)
    //
    // For k_iter (which K/16 slice):
    //   qs_lo = thread (k_iter*2) has elements k_iter*16 + {0..7}
    //   qs_hi = thread (k_iter*2+1) has elements k_iter*16 + {8..15}
    //
    // Q5_1 dequant: w = d * q5 + m where q5 is 0-31 (asymmetric)
    
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
        
        // Q5_1 K/128 byte layout (112 bytes):
        // Based on struct: qs0-3, qh0123, dm0, qs4-7, qh4567, dm1, dm2, qh891011, qs8-11, dm3, qs12-15, qh12131415
        //
        // qs byte offsets for each thread:
        static constexpr int qs_byte_offset[16] = {
            0, 4, 8, 12,      // threads 0-3 (qs0-3)
            24, 28, 32, 36,   // threads 4-7 (qs4-7)
            56, 60, 64, 68,   // threads 8-11 (qs8-11)
            76, 80, 84, 88    // threads 12-15 (qs12-15)
        };
        // qh byte offsets: qh0123=16, qh4567=40, qh891011=52, qh12131415=92
        static constexpr int qh_byte_offset[4] = {16, 40, 52, 92};
        // dm byte offsets: dm0=20, dm1=44, dm2=48, dm3=72
        static constexpr int dm_byte_offset[4] = {20, 44, 48, 72};
        
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
        // Q5_1 nibble layout in int (same as Q4/Q5_0):
        //   bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
        //   [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        //
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // k_group → h_base: 0→0, 1→2, 2→4, 3→6
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;  // Each k_group needs 2 adjacent h bits
        
        constexpr int LO_MASK_5BIT = 0x001f001f;
        
        // Build 5-bit values: low 4 bits from qs, high bit from qh
        const uint32_t lo_nibbles = ((uint32_t)qs_lo >> shift) & 0x000f000f;
        const uint32_t hi_nibbles = ((uint32_t)qs_hi >> shift) & 0x000f000f;
        
        // High bits: bit h_base goes to element 0, bit h_base+1 goes to element 1
        const uint32_t hb_lo = (((h_lo >> h_base) & 1) << 4) | (((h_lo >> (h_base + 1)) & 1) << 20);
        const uint32_t hb_hi = (((h_hi >> h_base) & 1) << 4) | (((h_hi >> (h_base + 1)) & 1) << 20);
        
        const uint32_t q5_lo = lo_nibbles | hb_lo;
        const uint32_t q5_hi = hi_nibbles | hb_hi;
        
        // =====================================================================
        // LOP3+HSUB with scale application
        // Q5_1: w = d * q5 + m (asymmetric quantization, no subtraction)
        // =====================================================================
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;  // Subtract 1024.0 to get raw q5 (0-31)
            const half2 d2 = __half2half2(__low2half(dm));
            const half2 m2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&lo_raw), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&hi_raw), *reinterpret_cast<const half2*>(&SUB));
            
            frag[0] = __hfma2(d2, raw0, m2);  // d*q5 + m
            frag[1] = __hfma2(d2, raw1, m2);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;  // Subtract 128.0 to get raw q5 (0-31)
            const half d_h = __low2half(dm);
            const half m_h = __high2half(dm);
            const __nv_bfloat162 d2 = __bfloat162bfloat162(__float2bfloat16(__half2float(d_h)));
            const __nv_bfloat162 m2 = __bfloat162bfloat162(__float2bfloat16(__half2float(m_h)));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX_BF);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX_BF);
            __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&lo_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&hi_raw), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            
            frag[0] = __hfma2(d2, raw0, m2);
            frag[1] = __hfma2(d2, raw1, m2);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            const half2 d2 = __half2half2(__low2half(dm));
            const half2 m2 = __half2half2(__high2half(dm));
            
            int lo_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_lo, LO_MASK_5BIT, EX);
            int hi_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_hi, LO_MASK_5BIT, EX);
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
    // Processes 4 consecutive k_iters with vector loads and hoisted computations.
    // Key optimizations:
    // - Single shift/h_base computation (constant across all k_iters)
    // - Vector loads where alignment permits: int4 (16B), int2 (8B), int (4B)
    // - Shared qh/dm loads for k_iter pairs (same scale_group)
    // =========================================================================
    
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        // =====================================================================
        // LANE-CONSTANT COMPUTATIONS (hoisted from loop)
        // =====================================================================
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Shift pattern: k_group → shift: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        const int h_base = k_group * 2;
        
        // LOP3 constants
        constexpr int LO_MASK_5BIT = 0x001f001f;
        constexpr int EX_FP16 = 0x64006400;
        constexpr uint32_t SUB_FP16 = 0x64006400;
        constexpr int EX_BF16 = 0x43004300;
        constexpr uint32_t SUB_BF16 = 0x43004300;
        
        if constexpr (half_idx == 0) {
            // =================================================================
            // HALF 0: k_iter 0,1,2,3 → threads 0-7
            // =================================================================
            // qs: threads 0-3 @ offsets 0,4,8,12 (16B aligned → int4)
            //     threads 4-7 @ offsets 24,28,32,36 (8B aligned → 2× int2)
            
            const int4 qs_01 = *reinterpret_cast<const int4*>(row_base + 0);   // qs[0-3]
            const int2 qs_2a = *reinterpret_cast<const int2*>(row_base + 24);  // qs[4-5]
            const int2 qs_2b = *reinterpret_cast<const int2*>(row_base + 32);  // qs[6-7]
            
            const int qh_01 = *reinterpret_cast<const int*>(row_base + 16);
            const half2 dm_01 = *reinterpret_cast<const half2*>(row_base + 20);
            const int qh_23 = *reinterpret_cast<const int*>(row_base + 40);
            const half2 dm_23 = *reinterpret_cast<const half2*>(row_base + 44);
            
            const uint8_t h0 = qh_01 & 0xFF, h1 = (qh_01 >> 8) & 0xFF;
            const uint8_t h2 = (qh_01 >> 16) & 0xFF, h3 = (qh_01 >> 24) & 0xFF;
            const uint8_t h4 = qh_23 & 0xFF, h5 = (qh_23 >> 8) & 0xFF;
            const uint8_t h6 = (qh_23 >> 16) & 0xFF, h7 = (qh_23 >> 24) & 0xFF;
            
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 d2_01 = __half2half2(__low2half(dm_01));
                const half2 m2_01 = __half2half2(__high2half(dm_01));
                const half2 d2_23 = __half2half2(__low2half(dm_23));
                const half2 m2_23 = __half2half2(__high2half(dm_23));
                
                // k_iter=0: threads 0,1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_01, w0, m2_01);
                    half2 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=1: threads 2,3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_01, w0, m2_01);
                    half2 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=2: threads 4,5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_23, w0, m2_23);
                    half2 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=3: threads 6,7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_23, w0, m2_23);
                    half2 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 d2_01 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_01))));
                const __nv_bfloat162 m2_01 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_01))));
                const __nv_bfloat162 d2_23 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_23))));
                const __nv_bfloat162 m2_23 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_23))));
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_01, w0, m2_01);
                    __nv_bfloat162 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_01, w0, m2_01);
                    __nv_bfloat162 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_23, w0, m2_23);
                    __nv_bfloat162 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_23, w0, m2_23);
                    __nv_bfloat162 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 d2_01 = __half2half2(__low2half(dm_01));
                const half2 m2_01 = __half2half2(__high2half(dm_01));
                const half2 d2_23 = __half2half2(__low2half(dm_23));
                const half2 m2_23 = __half2half2(__high2half(dm_23));
                
                // k_iter=0
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h0 >> h_base) & 1) << 4) | (((h0 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h1 >> h_base) & 1) << 4) | (((h1 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_01, w0, m2_01);
                    half2 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=1
                {
                    const uint32_t lo_nib = ((uint32_t)qs_01.z >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_01.w >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h2 >> h_base) & 1) << 4) | (((h2 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h3 >> h_base) & 1) << 4) | (((h3 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_01, w0, m2_01);
                    half2 f1 = __hfma2(d2_01, w1, m2_01);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=2
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h4 >> h_base) & 1) << 4) | (((h4 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h5 >> h_base) & 1) << 4) | (((h5 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_23, w0, m2_23);
                    half2 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=3
                {
                    const uint32_t lo_nib = ((uint32_t)qs_2b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_2b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h6 >> h_base) & 1) << 4) | (((h6 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h7 >> h_base) & 1) << 4) | (((h7 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_23, w0, m2_23);
                    half2 f1 = __hfma2(d2_23, w1, m2_23);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
            }
            
        } else {
            // =================================================================
            // HALF 1: k_iter 4,5,6,7 → threads 8-15
            // =================================================================
            // qs: threads 8-11 @ offsets 56,60,64,68 (8B aligned → 2× int2)
            //     threads 12-15 @ offsets 76,80,84,88 (4B aligned → 4× int)
            
            const int2 qs_4a = *reinterpret_cast<const int2*>(row_base + 56);  // qs[8-9]
            const int2 qs_4b = *reinterpret_cast<const int2*>(row_base + 64);  // qs[10-11]
            const int qs_12 = *reinterpret_cast<const int*>(row_base + 76);
            const int qs_13 = *reinterpret_cast<const int*>(row_base + 80);
            const int qs_14 = *reinterpret_cast<const int*>(row_base + 84);
            const int qs_15 = *reinterpret_cast<const int*>(row_base + 88);
            
            const int qh_45 = *reinterpret_cast<const int*>(row_base + 52);
            const half2 dm_45 = *reinterpret_cast<const half2*>(row_base + 48);
            const int qh_67 = *reinterpret_cast<const int*>(row_base + 92);
            const half2 dm_67 = *reinterpret_cast<const half2*>(row_base + 72);
            
            const uint8_t h8 = qh_45 & 0xFF, h9 = (qh_45 >> 8) & 0xFF;
            const uint8_t h10 = (qh_45 >> 16) & 0xFF, h11 = (qh_45 >> 24) & 0xFF;
            const uint8_t h12 = qh_67 & 0xFF, h13 = (qh_67 >> 8) & 0xFF;
            const uint8_t h14 = (qh_67 >> 16) & 0xFF, h15 = (qh_67 >> 24) & 0xFF;
            
            if constexpr (std::is_same_v<compute_t, half>) {
                const half2 d2_45 = __half2half2(__low2half(dm_45));
                const half2 m2_45 = __half2half2(__high2half(dm_45));
                const half2 d2_67 = __half2half2(__low2half(dm_67));
                const half2 m2_67 = __half2half2(__high2half(dm_67));
                
                // k_iter=4: threads 8,9
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_45, w0, m2_45);
                    half2 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=5: threads 10,11
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_45, w0, m2_45);
                    half2 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=6: threads 12,13
                {
                    const uint32_t lo_nib = ((uint32_t)qs_12 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_13 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_67, w0, m2_67);
                    half2 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=7: threads 14,15
                {
                    const uint32_t lo_nib = ((uint32_t)qs_14 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_15 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_67, w0, m2_67);
                    half2 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
                const __nv_bfloat162 d2_45 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_45))));
                const __nv_bfloat162 m2_45 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_45))));
                const __nv_bfloat162 d2_67 = __bfloat162bfloat162(__float2bfloat16(__half2float(__low2half(dm_67))));
                const __nv_bfloat162 m2_67 = __bfloat162bfloat162(__float2bfloat16(__half2float(__high2half(dm_67))));
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_45, w0, m2_45);
                    __nv_bfloat162 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_45, w0, m2_45);
                    __nv_bfloat162 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_12 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_13 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_67, w0, m2_67);
                    __nv_bfloat162 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_14 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_15 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_BF16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_BF16);
                    __nv_bfloat162 w0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw0), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 w1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&raw1), *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF16));
                    __nv_bfloat162 f0 = __hfma2(d2_67, w0, m2_67);
                    __nv_bfloat162 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
                
            } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
                const half2 d2_45 = __half2half2(__low2half(dm_45));
                const half2 m2_45 = __half2half2(__high2half(dm_45));
                const half2 d2_67 = __half2half2(__low2half(dm_67));
                const half2 m2_67 = __half2half2(__high2half(dm_67));
                
                // k_iter=4
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4a.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4a.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h8 >> h_base) & 1) << 4) | (((h8 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h9 >> h_base) & 1) << 4) | (((h9 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_45, w0, m2_45);
                    half2 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[0] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[1] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=5
                {
                    const uint32_t lo_nib = ((uint32_t)qs_4b.x >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_4b.y >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h10 >> h_base) & 1) << 4) | (((h10 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h11 >> h_base) & 1) << 4) | (((h11 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_45, w0, m2_45);
                    half2 f1 = __hfma2(d2_45, w1, m2_45);
                    frag_b[2] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[3] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=6
                {
                    const uint32_t lo_nib = ((uint32_t)qs_12 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_13 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h12 >> h_base) & 1) << 4) | (((h12 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h13 >> h_base) & 1) << 4) | (((h13 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_67, w0, m2_67);
                    half2 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[4] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[5] = *reinterpret_cast<uint32_t*>(&f1);
                }
                // k_iter=7
                {
                    const uint32_t lo_nib = ((uint32_t)qs_14 >> shift) & 0x000f000f;
                    const uint32_t hi_nib = ((uint32_t)qs_15 >> shift) & 0x000f000f;
                    const uint32_t hb_lo = (((h14 >> h_base) & 1) << 4) | (((h14 >> (h_base + 1)) & 1) << 20);
                    const uint32_t hb_hi = (((h15 >> h_base) & 1) << 4) | (((h15 >> (h_base + 1)) & 1) << 20);
                    int raw0 = lop3<0xEA>(lo_nib | hb_lo, LO_MASK_5BIT, EX_FP16);
                    int raw1 = lop3<0xEA>(hi_nib | hb_hi, LO_MASK_5BIT, EX_FP16);
                    half2 w0 = __hsub2(*reinterpret_cast<half2*>(&raw0), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 w1 = __hsub2(*reinterpret_cast<half2*>(&raw1), *reinterpret_cast<const half2*>(&SUB_FP16));
                    half2 f0 = __hfma2(d2_67, w0, m2_67);
                    half2 f1 = __hfma2(d2_67, w1, m2_67);
                    frag_b[6] = *reinterpret_cast<uint32_t*>(&f0);
                    frag_b[7] = *reinterpret_cast<uint32_t*>(&f1);
                }
            }
        }
    }
};

// Convenience aliases for Q5_1
using Q51_Dequant_FP16 = gemx_dequant_traits<block_c_q5_1, half, half>;
using Q51_Dequant_BF16 = gemx_dequant_traits<block_c_q5_1, __nv_bfloat16, __nv_bfloat16>;
using Q51_Dequant_FP8 = gemx_dequant_traits<block_c_q5_1, __nv_fp8_e4m3, half>;