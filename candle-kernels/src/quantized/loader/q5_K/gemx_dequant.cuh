// The Q5_K MMA dequantization traits, including the byte-permuted Q5_KO twin
// and its wavefront-optimal chunk.
//
// `gemx_dequant_traits` maps the Q5_K MMA interface onto compile-time lane
// dispatch. The Q5_KO specializations are the byte-permuted twin, and its
// lane-major K/1024 chunk reads the 640-byte quant region in wavefront order.
// The `Q5K_Dequant_*` aliases name the element types.

// =============================================================================
// MMA INTERFACE - COMPILE-TIME LANE DISPATCH FOR Q5_K
// =============================================================================
//
// Q5_K K/128 block layout (112 bytes = 28 ints):
//   data[0]  = dm0 (half2: scale, neg_min for threads 0-3)
//   data[1]  = qh0123 (packed uint8_t high bits for threads 0-3)
//   data[2-9] = qs0-qs7 (threads 0-7)
//   data[10] = qh4567
//   data[11] = dm1
//   data[12] = dm2
//   data[13] = qh891011
//   data[14-21] = qs8-qs15 (threads 8-15)
//   data[22] = qh12131415
//   data[23] = dm3
//
// Each qs contains 8 × 4-bit nibbles. Each qh byte contains 8 × 1-bit high bits.
// Combined: q5 = nibble | (high_bit << 4)
//
// =============================================================================

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_K, compute_t, scale_t> {
    static constexpr int K128_BYTES = 112;

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 5-bit = 4-bit nibble (qs) + 1 high bit (qh), affine, per-32 scale (dm0..3 each
    // cover 32 K, aligning with the k32 MMA sub). Same unpack as Q5_1 (mask + prmt
    // 0x3120 nibbles; high bit spread into byte bit 4). Fold applies d·C + m·Σx with
    // the explicit affine {scale, neg_min}.
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_OFF[16] =
            {8, 12, 16, 20, 24, 28, 32, 36, 56, 60, 64, 68, 72, 76, 80, 84};
        constexpr int QH_BASE[4] = {4, 40, 52, 88};
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
    // Per-sub affine {scale (low), neg_min (high)} from dm0..dm3 at data[0,11,12,23].
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {0, 44, 48, 92};
        return *reinterpret_cast<const half2*>(row_block + DM_OFF[sub]);
    }

    using Frags = GemxFragmentTypes<compute_t>;
    using FragB = typename Frags::FragB;
    
    // Compile-time lane parameter extraction
    template <int LANE>
    struct lane_params {
        static constexpr int N = LANE / 4;              // Row 0-7
        static constexpr int K_GROUP = LANE % 4;        // K-group 0-3
        static constexpr int THREAD_IN_BLOCK = K_GROUP / 2;  // 0 for K_GROUP 0-1, 1 for 2-3
        static constexpr int NIBBLE_HALF = K_GROUP % 2;  // Which 4 nibbles: 0=first, 1=second
    };
    
    // Byte offset tables for Q5_K layout
    // qs indices: 2-9 for threads 0-7, 14-21 for threads 8-15
    static constexpr int qs_data_idx[16] = {
        2, 3, 4, 5, 6, 7, 8, 9,     // threads 0-7
        14, 15, 16, 17, 18, 19, 20, 21  // threads 8-15
    };
    
    // qh word indices: qh0123=1, qh4567=10, qh891011=13, qh12131415=22
    static constexpr int qh_data_idx[4] = {1, 10, 13, 22};
    
    // dm indices: dm0=0, dm1=11, dm2=12, dm3=23
    static constexpr int dm_data_idx[4] = {0, 11, 12, 23};
    
    // -------------------------------------------------------------------------
    // BUILD Q5 ARRAYS: Combine nibbles with high bits
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static void build_q5_arrays(
        int ql, uint8_t qh, uint32_t& q5_even, uint32_t& q5_odd
    ) {
        // Extract nibbles from LOP3-ready layout
        // ql bits: [3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6, [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        // Extract high bits from qh: h0(bit0), h1(bit1), ..., h7(bit7)
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // Get nibble pairs in consecutive order
        const uint32_t nib_01 = (ql >> 0) & LO_MASK;   // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;   // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;   // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;  // (n6, n7)
        
        // Extract and position high bits to OR with nibbles (shift by 4)
        const uint32_t h0 = ((qh >> 0) & 1) << 4;
        const uint32_t h1 = ((qh >> 1) & 1) << 20;  // Goes into high half-word
        const uint32_t h2 = ((qh >> 2) & 1) << 4;
        const uint32_t h3 = ((qh >> 3) & 1) << 20;
        const uint32_t h4 = ((qh >> 4) & 1) << 4;
        const uint32_t h5 = ((qh >> 5) & 1) << 20;
        const uint32_t h6 = ((qh >> 6) & 1) << 4;
        const uint32_t h7 = ((qh >> 7) & 1) << 20;
        
        // Combine: q5[i] = nibble[i] | (high_bit[i] << 4)
        // Build q5_even: byte0=q5_0, byte1=q5_2, byte2=q5_4, byte3=q5_6
        // Build q5_odd:  byte0=q5_1, byte1=q5_3, byte2=q5_5, byte3=q5_7
        const uint32_t q5_01 = nib_01 | h0 | h1;
        const uint32_t q5_23 = nib_23 | h2 | h3;
        const uint32_t q5_45 = nib_45 | h4 | h5;
        const uint32_t q5_67 = nib_67 | h6 | h7;
        
        // Rearrange to even/odd for PRMT
        q5_even = (q5_01 & 0x1f) | ((q5_23 & 0x1f) << 8) | ((q5_45 & 0x1f) << 16) | ((q5_67 & 0x1f) << 24);
        q5_odd = ((q5_01 >> 16) & 0x1f) | (((q5_23 >> 16) & 0x1f) << 8) | 
                 (((q5_45 >> 16) & 0x1f) << 16) | (((q5_67 >> 16) & 0x1f) << 24);
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS from q5_even/q5_odd arrays
    // -------------------------------------------------------------------------
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(
        uint32_t q5_even, uint32_t q5_odd, FragB_t& frag
    ) {
        constexpr int LO_MASK = 0x001f001f;  // 5-bit mask
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            // NIBBLE_HALF=0: pairs 0,1 (q5_0, q5_1, q5_2, q5_3)
            // NIBBLE_HALF=1: pairs 2,3 (q5_4, q5_5, q5_6, q5_7)
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q5_even, q5_odd) : 
                prmt_build_lop3_pair_2(q5_even, q5_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q5_even, q5_odd) : 
                prmt_build_lop3_pair_3(q5_even, q5_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43004300;
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q5_even, q5_odd) : 
                prmt_build_lop3_pair_2(q5_even, q5_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q5_even, q5_odd) : 
                prmt_build_lop3_pair_3(q5_even, q5_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64006400;
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q5_even, q5_odd) : 
                prmt_build_lop3_pair_2(q5_even, q5_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q5_even, q5_odd) : 
                prmt_build_lop3_pair_3(q5_even, q5_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX_FP16);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX_FP16);
            
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
    // EXTRACT 8 ELEMENTS for K=32 MMA (FP8)
    // -------------------------------------------------------------------------
    template <typename FragB_t>
    __device__ __forceinline__ static void extract_8_elements(
        uint32_t q5_even, uint32_t q5_odd, FragB_t& frag0, FragB_t& frag1
    ) {
        extract_4_elements<0>(q5_even, q5_odd, frag0);
        extract_4_elements<1>(q5_even, q5_odd, frag1);
    }
    
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // MMA B fragment layout for m16n8k16:
    //   frag[0] = half2(K0, K1) or (K2,K3) etc based on k_group
    //   frag[1] = half2(K8, K9) or (K10,K11) etc based on k_group
    // So frag[0] comes from first qs of K/16 tile, frag[1] from second qs.
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // OPTIMIZED ADDRESS COMPUTATION - All bit operations
        // =====================================================================
        const int row = lane >> 2;          // N: 0-7
        const int k_group = lane & 3;       // 0-3
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        
        // Local copies of index tables to avoid device code ODR issues
        static constexpr int local_qs_data_idx[16] = {
            2, 3, 4, 5, 6, 7, 8, 9,
            14, 15, 16, 17, 18, 19, 20, 21
        };
        static constexpr int local_qh_data_idx[4] = {1, 10, 13, 22};
        static constexpr int local_dm_data_idx[4] = {0, 11, 12, 23};
        
        const int* data_ptr = reinterpret_cast<const int*>(row_base);
        
        // Load BOTH qs entries for this K/16 tile
        const int qs_idx_lo = (k_iter << 1);      // k_iter * 2 for K=0-7
        const int qs_idx_hi = (k_iter << 1) | 1;  // k_iter * 2 + 1 for K=8-15
        
        const int ql_lo = data_ptr[local_qs_data_idx[qs_idx_lo]];
        const int ql_hi = data_ptr[local_qs_data_idx[qs_idx_hi]];
        
        // Get qh for both (they share the same qh word within a group of 4)
        const uint32_t qh_word_lo = static_cast<uint32_t>(data_ptr[local_qh_data_idx[qs_idx_lo / 4]]);
        const uint32_t qh_word_hi = static_cast<uint32_t>(data_ptr[local_qh_data_idx[qs_idx_hi / 4]]);
        const uint8_t qh_lo = (qh_word_lo >> ((qs_idx_lo % 4) * 8)) & 0xFF;
        const uint8_t qh_hi = (qh_word_hi >> ((qs_idx_hi % 4) * 8)) & 0xFF;
        
        // dm from first qs (both qs in a K/16 tile share the same scale group)
        const half2 dm = *reinterpret_cast<const half2*>(&data_ptr[local_dm_data_idx[qs_idx_lo / 4]]);
        
        const half scale_h = __low2half(dm);
        const half neg_min_h = __high2half(dm);
        
        // =====================================================================
        // OPTIMIZED Q5 EXTRACTION - Compute ONLY the k_group-specific pair
        // =====================================================================
        // Instead of computing all 8 q5 pairs and selecting 2 with ternaries,
        // we compute ONLY the 2 pairs we need based on k_group.
        //
        // k_group determines which nibble pair we need:
        //   k_group=0: nib_01 (shift 0)   → q5 elements 0,1
        //   k_group=1: nib_23 (shift 8)   → q5 elements 2,3
        //   k_group=2: nib_45 (shift 4)   → q5 elements 4,5
        //   k_group=3: nib_67 (shift 12)  → q5 elements 6,7
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // =====================================================================
        // OPTIMIZED SHIFT COMPUTATION - Pure bit operations
        // =====================================================================
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // Formula: shift = ((k_group & 1) << 3) | ((k_group & 2) << 1)
        const int nib_shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        // Extract only the nibble pair we need
        const uint32_t nib_lo = (ql_lo >> nib_shift) & LO_MASK;
        const uint32_t nib_hi = (ql_hi >> nib_shift) & LO_MASK;
        
        // High bit extraction: k_group determines which 2 bits we need
        // For k_group g, we need high bits at positions (2*g, 2*g+1)
        const int h_shift = k_group * 2;
        const uint32_t h_lo_even = ((qh_lo >> h_shift) & 1) << 4;
        const uint32_t h_lo_odd = ((qh_lo >> (h_shift + 1)) & 1) << 20;
        const uint32_t h_hi_even = ((qh_hi >> h_shift) & 1) << 4;
        const uint32_t h_hi_odd = ((qh_hi >> (h_shift + 1)) & 1) << 20;
        
        // Combine nibbles with high bits to form q5 pairs (already in half-word format)
        const uint32_t p_lo = nib_lo | h_lo_even | h_lo_odd;
        const uint32_t p_hi = nib_hi | h_hi_even | h_hi_odd;
        
        // Single LOP3+HSUB path (no duplication!)
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int LO_MASK = 0x001f001f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK, EX);
            const half2 raw_lo = __hsub2(*reinterpret_cast<half2*>(&w_lo), *reinterpret_cast<const half2*>(&SUB));
            const half2 raw_hi = __hsub2(*reinterpret_cast<half2*>(&w_hi), *reinterpret_cast<const half2*>(&SUB));
            
            // Apply scale and neg_min: w = scale * raw + neg_min
            const half2 scale2 = __half2half2(scale_h);
            const half2 neg_min2 = __half2half2(neg_min_h);
            frag[0] = __hfma2(scale2, raw_lo, neg_min2);
            frag[1] = __hfma2(scale2, raw_hi, neg_min2);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int LO_MASK = 0x001f001f;
            constexpr int EX = 0x43004300;
            constexpr uint32_t SUB = 0x43004300;
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK, EX);
            const __nv_bfloat162 raw_lo = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_lo), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
            const __nv_bfloat162 raw_hi = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_hi), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
            
            // Apply scale and neg_min: w = scale * raw + neg_min
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(scale_h)));
            const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(__float2bfloat16(__half2float(neg_min_h)));
            frag[0] = __hfma2(scale2, raw_lo, neg_min2);
            frag[1] = __hfma2(scale2, raw_hi, neg_min2);
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int LO_MASK = 0x001f001f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK, EX);
            const half2 raw_lo = __hsub2(*reinterpret_cast<half2*>(&w_lo), *reinterpret_cast<const half2*>(&SUB));
            const half2 raw_hi = __hsub2(*reinterpret_cast<half2*>(&w_hi), *reinterpret_cast<const half2*>(&SUB));
            
            // Apply scale and neg_min: w = scale * raw + neg_min
            const half2 scale2 = __half2half2(scale_h);
            const half2 neg_min2 = __half2half2(neg_min_h);
            half2 w0 = __hfma2(scale2, raw_lo, neg_min2);
            half2 w1 = __hfma2(scale2, raw_hi, neg_min2);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY OPTIMIZED)
    // =========================================================================
    // Optimizations applied:
    // 1. Vector loads: 2× int4 for ql (8 values), instead of 8× scalar loads
    // 2. Single shift computation for k_group, amortized across all slices
    // 3. Inline high-bit extraction (pure ALU, no function calls)
    // 4. Shared scales: slices 0-1 share dm0, slices 2-3 share dm1
    // 5. No loop/function call overhead - fully unrolled
    //
    // Q5_K K/128 layout (112 bytes):
    //   half_idx=0: ql at data[2-9], qh at data[1,10], dm at data[0,11]
    //   half_idx=1: ql at data[14-21], qh at data[13,22], dm at data[12,23]
    // =========================================================================
    template <int half_idx>
    __device__ __forceinline__ static void dequant_for_4x_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int lane,
        uint32_t* frag_b
    ) {
        static_assert(half_idx == 0 || half_idx == 1, "half_idx must be 0 or 1");
        
        const int row = lane >> 2;
        const int k_group = lane & 3;
        const uint8_t* row_base = smem_rows + row * K128_BYTES;
        const int* data_ptr = reinterpret_cast<const int*>(row_base);
        
        // Nibble shift for k_group: 0→0, 1→8, 2→4, 3→12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        // High bit shift: k_group determines which 2 bits we need from each qh byte
        const int h_shift = k_group * 2;
        
        // Compile-time offsets based on half_idx
        constexpr int ql_base = (half_idx == 0) ? 2 : 14;
        constexpr int qh0_idx = (half_idx == 0) ? 1 : 13;   // qh for slices 0,1
        constexpr int qh1_idx = (half_idx == 0) ? 10 : 22;  // qh for slices 2,3
        constexpr int dm0_idx = (half_idx == 0) ? 0 : 12;   // dm for slices 0,1
        constexpr int dm1_idx = (half_idx == 0) ? 11 : 23;  // dm for slices 2,3
        
        // Load all 8 ql values as 4× int2 (64-bit aligned vector loads)
        // ql_base offset is 8 or 56 bytes from row start (8-byte aligned, not 16-byte)
        const int2* ql_ptr = reinterpret_cast<const int2*>(data_ptr + ql_base);
        const int2 ql_01 = ql_ptr[0];
        const int2 ql_23 = ql_ptr[1];
        const int2 ql_45 = ql_ptr[2];
        const int2 ql_67 = ql_ptr[3];
        
        // Load qh words and dm scales
        const uint32_t qh0 = static_cast<uint32_t>(data_ptr[qh0_idx]);
        const uint32_t qh1 = static_cast<uint32_t>(data_ptr[qh1_idx]);
        const half2 dm0 = *reinterpret_cast<const half2*>(data_ptr + dm0_idx);
        const half2 dm1 = *reinterpret_cast<const half2*>(data_ptr + dm1_idx);
        
        // Extract scales and neg_mins
        const half sc01 = __low2half(dm0), neg_min01 = __high2half(dm0);
        const half sc23 = __low2half(dm1), neg_min23 = __high2half(dm1);
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // Extract nibble pairs for all slices (shifted for k_group)
        const uint32_t nib0_lo = (ql_01.x >> shift) & LO_MASK;
        const uint32_t nib0_hi = (ql_01.y >> shift) & LO_MASK;
        const uint32_t nib1_lo = (ql_23.x >> shift) & LO_MASK;
        const uint32_t nib1_hi = (ql_23.y >> shift) & LO_MASK;
        const uint32_t nib2_lo = (ql_45.x >> shift) & LO_MASK;
        const uint32_t nib2_hi = (ql_45.y >> shift) & LO_MASK;
        const uint32_t nib3_lo = (ql_67.x >> shift) & LO_MASK;
        const uint32_t nib3_hi = (ql_67.y >> shift) & LO_MASK;
        
        // Extract high bit pairs inline (pure ALU, no LUT)
        // Each qh byte has 8 high bits; h_shift selects which 2 we need
        #define HB_PAIR(qh_byte) \
            ((((qh_byte) >> h_shift) & 1) << 4) | ((((qh_byte) >> (h_shift + 1)) & 1) << 20)
        
        const uint32_t hb0_lo = HB_PAIR((qh0 >> 0) & 0xFF);
        const uint32_t hb0_hi = HB_PAIR((qh0 >> 8) & 0xFF);
        const uint32_t hb1_lo = HB_PAIR((qh0 >> 16) & 0xFF);
        const uint32_t hb1_hi = HB_PAIR((qh0 >> 24) & 0xFF);
        const uint32_t hb2_lo = HB_PAIR((qh1 >> 0) & 0xFF);
        const uint32_t hb2_hi = HB_PAIR((qh1 >> 8) & 0xFF);
        const uint32_t hb3_lo = HB_PAIR((qh1 >> 16) & 0xFF);
        const uint32_t hb3_hi = HB_PAIR((qh1 >> 24) & 0xFF);
        
        #undef HB_PAIR
        
        // Combine nibbles with high bits to form q5 pairs
        const uint32_t p0_lo = nib0_lo | hb0_lo, p0_hi = nib0_hi | hb0_hi;
        const uint32_t p1_lo = nib1_lo | hb1_lo, p1_hi = nib1_hi | hb1_hi;
        const uint32_t p2_lo = nib2_lo | hb2_lo, p2_hi = nib2_hi | hb2_hi;
        const uint32_t p3_lo = nib3_lo | hb3_lo, p3_hi = nib3_hi | hb3_hi;
        
        // Type-specific dequant using LOP3 + hsub2 + hfma2
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int LO_MASK_5 = 0x001f001f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            const half2 sc01_2 = __half2half2(sc01), nm01_2 = __half2half2(neg_min01);
            const half2 sc23_2 = __half2half2(sc23), nm23_2 = __half2half2(neg_min23);
            
            // Slice 0 (uses dm0)
            int w0 = lop3<0xEA>(p0_lo, LO_MASK_5, EX);
            int w1 = lop3<0xEA>(p0_hi, LO_MASK_5, EX);
            half2 r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            half2 r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 1 (uses dm0)
            w0 = lop3<0xEA>(p1_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p1_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 2 (uses dm1)
            w0 = lop3<0xEA>(p2_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p2_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 3 (uses dm1)
            w0 = lop3<0xEA>(p3_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p3_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int LO_MASK_5 = 0x001f001f;
            constexpr int EX = 0x43004300;
            constexpr uint32_t SUB = 0x43004300;
            
            const __nv_bfloat162 sc01_2 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc01)));
            const __nv_bfloat162 nm01_2 = __bfloat162bfloat162(__float2bfloat16(__half2float(neg_min01)));
            const __nv_bfloat162 sc23_2 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc23)));
            const __nv_bfloat162 nm23_2 = __bfloat162bfloat162(__float2bfloat16(__half2float(neg_min23)));
            
            int w0 = lop3<0xEA>(p0_lo, LO_MASK_5, EX);
            int w1 = lop3<0xEA>(p0_hi, LO_MASK_5, EX);
            __nv_bfloat162 r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm01_2);
            __nv_bfloat162 r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm01_2);
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p1_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p1_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm01_2);
            r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm01_2);
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p2_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p2_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm23_2);
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p3_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p3_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)), nm23_2);
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else {
            // FP8: use FP16 path
            constexpr int LO_MASK_5 = 0x001f001f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64006400;
            
            const half2 sc01_2 = __half2half2(sc01), nm01_2 = __half2half2(neg_min01);
            const half2 sc23_2 = __half2half2(sc23), nm23_2 = __half2half2(neg_min23);
            
            int w0 = lop3<0xEA>(p0_lo, LO_MASK_5, EX);
            int w1 = lop3<0xEA>(p0_hi, LO_MASK_5, EX);
            half2 r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            half2 r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p1_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p1_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            r1 = __hfma2(sc01_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm01_2);
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p2_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p2_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>(p3_lo, LO_MASK_5, EX);
            w1 = lop3<0xEA>(p3_hi, LO_MASK_5, EX);
            r0 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            r1 = __hfma2(sc23_2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)), nm23_2);
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
        }
    }
};

// Q5_KO: byte-permuted twin of Q5_K — qs contiguous (field m at m*4), qh ints
// contiguous (64-79), the four (scale,-min) pairs grouped (80-95). Inherits Q5_K and
// overrides only the two int8 accessors with the regularized offsets; identical
// nibble + 5th-bit-spread math. Int8-only (FP accessors inherited, not built).
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_KO, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q5_K, compute_t, scale_t> {
    using base = gemx_dequant_traits<block_c_q5_K, compute_t, scale_t>;

    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        const int row = lane >> 2;
        const int q3 = lane & 3;
        // De-interleaved Q5_KO block is 80 B (quant only) — NOT base::K128_BYTES (112,
        // the interleaved Q5_K size). Scales live in the separate region (sub_dm unused).
        const uint8_t* rb = warp_rows + row * smem_row_stride<block_c_q5_KO_k128>::value;
        const int sh = (q3 & 1) * 4;
        // qs ints interleaved [I0,I2,I1,I3] per sub so the lane's two qs ints are
        // adjacent → one int2 (LDS.64) instead of two int loads (−1 LDS/sub: helps
        // push Q5 back under the MIO-throttle cliff). q3<2 → {I0,I2}; q3>=2 → {I1,I3}.
        const int2 qv = *reinterpret_cast<const int2*>(rb + sub * 16 + (q3 >> 1) * 8);
        const uint32_t nib0 = __byte_perm((qv.x >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t nib1 = __byte_perm((qv.y >> sh) & 0x0F0F0F0F, 0, 0x3120);
        // qh0/qh1 are bytes (q3>>1) and (q3>>1)+2 of the SAME qh int (qh[sub] at
        // 64+sub*4, since m0>>2 == m1>>2 == sub). Load that int once (broadcast across
        // the k-group) and extract the two bytes from registers — replacing two slow
        // sub-word byte loads with one aligned int load: the Q5 qh bottleneck.
        const uint32_t qhw = *reinterpret_cast<const uint32_t*>(rb + 64 + sub * 4);
        const uint32_t qh0 = (qhw >> ((q3 >> 1) * 8)) & 0xFFu;
        const uint32_t qh1 = (qhw >> (((q3 >> 1) + 2) * 8)) & 0xFFu;
        const uint32_t hb0 = (qh0 >> sh) & 0xF;
        const uint32_t hb1 = (qh1 >> sh) & 0xF;
        b_frag[0] = nib0 | (((hb0 * 0x00204081u) & 0x01010101u) << 4);
        b_frag[1] = nib1 | (((hb1 * 0x00204081u) & 0x01010101u) << 4);
    }
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {80, 84, 88, 92};
        return *reinterpret_cast<const half2*>(row_block + DM_OFF[sub]);
    }
};

// Q5_KO K/1024 chunk — WAVEFRONT-OPTIMAL dequant, LANE-MAJOR. The 640 B quant region splits
// into a 512 B ql stream (lane's 4 subs at lane*16+sub*4, like Q4) and a 128 B 5th-bit stream
// (lane's 4 subs' bytes at 512+lane*4+sub; low nibble = the four 5th-bits of b_frag[0], high =
// those of b_frag[1]). ONE int4 LDS at lane*16 pulls all 4 subs' ql, ONE uint32 LDS at
// 512+lane*4 pulls all 4 subs' hi-bytes — both conflict-free (lane*16 / lane*4 → distinct
// banks per quarter-warp phase). The 5th bits spread to bit 4 via the 0x00204081/0x01010101
// magic (Q5_K math). Values 0..31; per-32 (scale,min) fold. dm at 640.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q5_KO_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q5_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int4 vv = *reinterpret_cast<const int4*>(chunk + lane * 16);
        const uint32_t s4[4] = {(uint32_t)vv.x, (uint32_t)vv.y, (uint32_t)vv.z, (uint32_t)vv.w};
        const uint32_t hi4 = *reinterpret_cast<const uint32_t*>(chunk + 512 + lane * 4);
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t nib0 = s4[sub] & 0x0F0F0F0Fu;
            const uint32_t nib1 = (s4[sub] >> 4) & 0x0F0F0F0Fu;
            const uint32_t hbb = (hi4 >> (sub * 8)) & 0xFFu;
            const uint32_t hb0 = hbb & 0xFu;
            const uint32_t hb1 = (hbb >> 4) & 0xFu;
            b_frags[sub][0] = nib0 | (((hb0 * 0x00204081u) & 0x01010101u) << 4);
            b_frags[sub][1] = nib1 | (((hb1 * 0x00204081u) & 0x01010101u) << 4);
        }
    }
};

// Type aliases
using Q5K_Dequant_FP16 = gemx_dequant_traits<block_c_q5_K, half, half>;
using Q5K_Dequant_BF16 = gemx_dequant_traits<block_c_q5_K, __nv_bfloat16, __nv_bfloat16>;
using Q5K_Dequant_FP8 = gemx_dequant_traits<block_c_q5_K, __nv_fp8_e4m3, half>;
