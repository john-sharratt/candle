// The GEMX dequant traits for Q6_K, Q6_KO and the Q6_KO K/1024 chunk.
//
// The `gemx_dequant_traits` specializations for Q6_K in MMA form, for Q6_KO
// whose compact block is already in ordered form, and for the Q6_KO K/1024
// chunk, which dequantizes lane-major; plus the `Q6K_Dequant_FP16`,
// `Q6K_Dequant_BF16` and `Q6K_Dequant_FP8` aliases.

// =============================================================================
// GEMX DEQUANT TRAITS - Q6_K (MMA INTERFACE)
// =============================================================================
//
// Q6_K K/128 block layout (128 bytes = 32 ints):
// Each group of 4 ints covers 2 threads:
//   data[g*4+0] = ql for thread g*2   (low 4 bits of 8 elements)
//   data[g*4+1] = ql for thread g*2+1 (low 4 bits of 8 elements)
//   data[g*4+2] = qh_g*2 | qh_g*2+1   (high 2 bits packed as uint16_t)
//   data[g*4+3] = scale | _pad        (scale as half for threads g*2, g*2+1)
//
// Each ql contains 8 × 4-bit nibbles. Each qh (uint16_t) contains 8 × 2-bit crumbs.
// Combined: q6 = nibble | (crumb << 4)
// Q6_K is symmetric: result = scale * (q6 - 32)
//
// =============================================================================

template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q6_K, compute_t, scale_t> {
    static constexpr bool implemented = true;
    static constexpr bool has_min = false;
    static constexpr int K112_BYTES = 112;  // Compact 112-byte layout

    // -------------------------------------------------------------------------
    // INT8 TENSOR-CORE PATH
    // -------------------------------------------------------------------------
    // 6-bit = 4-bit nibble (ql) + 2-bit crumb (qh), symmetric (value = scale·(q6-32)).
    // Q6_K scales are per-16, but the k32 MMA folds one scale per 32-K sub; we re-bin
    // the two per-16 scales to one per-32 scale (average) in sub_dm. Feeding CENTERED
    // signed int8 (c = q6-32) makes the format symmetric, so neg_min = 0 and no
    // per-16 activation sum is needed — the existing single-scale fold applies.
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        const int row = lane >> 2;
        const int q3 = lane & 3;
        const uint8_t* rb = warp_rows + row * K112_BYTES;
        const int sh = (q3 & 1) * 4;
        const int m0 = sub * 4 + (q3 >> 1);
        const int m1 = m0 + 2;
        // ql nibbles (low 4 bits) → natural order via prmt 0x3120.
        const int v0 = *reinterpret_cast<const int*>(rb + m0 * 4);
        const int v1 = *reinterpret_cast<const int*>(rb + m1 * 4);
        const uint32_t nib0 = __byte_perm((v0 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t nib1 = __byte_perm((v1 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        // qh crumbs (2 bits): field m's 16-bit qh at byte 64 + (m>>1)*4 + (m&1)*2.
        const uint32_t qh0 = *reinterpret_cast<const uint16_t*>(rb + 64 + (m0 >> 1) * 4 + (m0 & 1) * 2);
        const uint32_t qh1 = *reinterpret_cast<const uint16_t*>(rb + 64 + (m1 >> 1) * 4 + (m1 & 1) * 2);
        const uint32_t cr0 = (qh0 >> ((q3 & 1) * 8)) & 0xFFu;
        const uint32_t cr1 = (qh1 >> ((q3 & 1) * 8)) & 0xFFu;
        // Spread crumb i (2 bits) into bits 4-5 of byte i → unsigned q6 in 0..63.
        uint32_t q6_0 = nib0 | ((cr0 & 0x03u) << 4) | ((cr0 & 0x0Cu) << 10)
                             | ((cr0 & 0x30u) << 16) | ((cr0 & 0xC0u) << 22);
        uint32_t q6_1 = nib1 | ((cr1 & 0x03u) << 4) | ((cr1 & 0x0Cu) << 10)
                             | ((cr1 & 0x30u) << 16) | ((cr1 & 0xC0u) << 22);
        // Center by -32: offset-binary → two's complement (flip bit 5, sign-extend 6-7).
        q6_0 ^= 0x20202020u; q6_0 |= ((q6_0 & 0x20202020u) << 1) | ((q6_0 & 0x20202020u) << 2);
        q6_1 ^= 0x20202020u; q6_1 |= ((q6_1 & 0x20202020u) << 1) | ((q6_1 & 0x20202020u) << 2);
        b_frag[0] = q6_0;
        b_frag[1] = q6_1;
    }
    // Two per-16 scales {scale_lo (K[0,16)), scale_hi (K[16,32))} for the split
    // 2-MMA fold (int8_scales_per_sub == 2). scales[2s] covers K-groups 4s,4s+1;
    // scales[2s+1] covers 4s+2,4s+3; they are contiguous halves at byte 96+4s.
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        return *reinterpret_cast<const half2*>(row_block + 96 + 4 * sub);
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
    
    // -------------------------------------------------------------------------
    // BUILD Q6 ARRAYS: Combine nibbles with crumbs
    // -------------------------------------------------------------------------
    __device__ __forceinline__ static void build_q6_arrays(
        int ql, uint16_t qh, uint32_t& q6_even, uint32_t& q6_odd
    ) {
        // Extract nibbles from LOP3-ready layout
        // ql bits: [3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6, [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        // qh contains crumbs: c0(bits 0-1), c1(bits 2-3), ..., c7(bits 14-15)
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // Get nibble pairs in consecutive order
        const uint32_t nib_01 = (ql >> 0) & LO_MASK;   // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;   // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;   // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;  // (n6, n7)
        
        // Extract and position crumbs to OR with nibbles (shift by 4)
        const uint32_t qh32 = static_cast<uint32_t>(qh);
        const uint32_t c0 = ((qh32 >> 0) & 0x3) << 4;
        const uint32_t c1 = ((qh32 >> 2) & 0x3) << 20;  // Goes into high half-word
        const uint32_t c2 = ((qh32 >> 4) & 0x3) << 4;
        const uint32_t c3 = ((qh32 >> 6) & 0x3) << 20;
        const uint32_t c4 = ((qh32 >> 8) & 0x3) << 4;
        const uint32_t c5 = ((qh32 >> 10) & 0x3) << 20;
        const uint32_t c6 = ((qh32 >> 12) & 0x3) << 4;
        const uint32_t c7 = ((qh32 >> 14) & 0x3) << 20;
        
        // Combine: q6[i] = nibble[i] | (crumb[i] << 4)
        const uint32_t q6_01 = nib_01 | c0 | c1;
        const uint32_t q6_23 = nib_23 | c2 | c3;
        const uint32_t q6_45 = nib_45 | c4 | c5;
        const uint32_t q6_67 = nib_67 | c6 | c7;
        
        // Rearrange to even/odd for PRMT
        q6_even = (q6_01 & 0x3f) | ((q6_23 & 0x3f) << 8) | ((q6_45 & 0x3f) << 16) | ((q6_67 & 0x3f) << 24);
        q6_odd = ((q6_01 >> 16) & 0x3f) | (((q6_23 >> 16) & 0x3f) << 8) | 
                 (((q6_45 >> 16) & 0x3f) << 16) | (((q6_67 >> 16) & 0x3f) << 24);
    }
    
    // -------------------------------------------------------------------------
    // EXTRACT 4 ELEMENTS from q6_even/q6_odd arrays (signed: subtract 32)
    // -------------------------------------------------------------------------
    template <int NIBBLE_HALF, typename FragB_t>
    __device__ __forceinline__ static void extract_4_elements(
        uint32_t q6_even, uint32_t q6_odd, FragB_t& frag
    ) {
        constexpr int LO_MASK = 0x003f003f;  // 6-bit mask
        
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64206420;  // half2(1024+32, 1024+32) for signed centering
            
            // NIBBLE_HALF=0: pairs 0,1 (q6_0, q6_1, q6_2, q6_3)
            // NIBBLE_HALF=1: pairs 2,3 (q6_4, q6_5, q6_6, q6_7)
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q6_even, q6_odd) : 
                prmt_build_lop3_pair_2(q6_even, q6_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q6_even, q6_odd) : 
                prmt_build_lop3_pair_3(q6_even, q6_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX);
            
            frag[0] = __hsub2(*reinterpret_cast<half2*>(&w01),
                              *reinterpret_cast<const half2*>(&SUB));
            frag[1] = __hsub2(*reinterpret_cast<half2*>(&w23),
                              *reinterpret_cast<const half2*>(&SUB));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t SUB_BF = 0x43204320;  // bf162(128+32, 128+32)
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q6_even, q6_odd) : 
                prmt_build_lop3_pair_2(q6_even, q6_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q6_even, q6_odd) : 
                prmt_build_lop3_pair_3(q6_even, q6_odd);
            
            int w01 = lop3<(0xf0 & 0xcc) | 0xaa>(p0, LO_MASK, EX_BF);
            int w23 = lop3<(0xf0 & 0xcc) | 0xaa>(p1, LO_MASK, EX_BF);
            
            frag[0] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w01),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
            frag[1] = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w23),
                              *reinterpret_cast<const __nv_bfloat162*>(&SUB_BF));
                              
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            constexpr int EX_FP16 = 0x64006400;
            constexpr uint32_t SUB_FP16 = 0x64206420;
            
            const uint32_t p0 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_0(q6_even, q6_odd) : 
                prmt_build_lop3_pair_2(q6_even, q6_odd);
            const uint32_t p1 = (NIBBLE_HALF == 0) ? 
                prmt_build_lop3_pair_1(q6_even, q6_odd) : 
                prmt_build_lop3_pair_3(q6_even, q6_odd);
            
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
        uint32_t q6_even, uint32_t q6_odd, FragB_t& frag0, FragB_t& frag1
    ) {
        extract_4_elements<0>(q6_even, q6_odd, frag0);
        extract_4_elements<1>(q6_even, q6_odd, frag1);
    }
    
    // =========================================================================
    // DEQUANT FOR MMA K=16 (FP16/BF16)
    // =========================================================================
    // =========================================================================
    // RUNTIME DEQUANT FOR MMA K=16 (for TC kernel with runtime k_iter, lane)
    // =========================================================================
    // COMPACT 112-BYTE LAYOUT: Contiguous arrays
    //   ql[16]    - bytes 0-63:   one int per thread
    //   qh[8]     - bytes 64-95:  one int per thread-pair (qh_lo | qh_hi << 16)
    //   scales[8] - bytes 96-111: one half per thread-pair
    //
    // MMA accesses k_iter 0-7, where each k_iter covers 16 elements (2 threads × 8 each)
    // Thread-pair for k_iter i: threads 2i, 2i+1
    
    __device__ __forceinline__ static void dequant_for_mma_k16_runtime(
        const uint8_t* __restrict__ smem_rows,
        int k_iter,
        int lane,
        FragB& frag
    ) {
        // =====================================================================
        // OPTIMIZED ADDRESS COMPUTATION - All bit operations
        // =====================================================================
        const int row = lane >> 2;          // N: 0-7 (which output row)
        const int k_group = lane & 3;       // 0-3 (which K pair within K/16)
        
        // Base pointer for this row's K/128 block
        const uint8_t* row_base = smem_rows + row * K112_BYTES;
        const block_c_q6_K_k128* blk = reinterpret_cast<const block_c_q6_K_k128*>(row_base);
        
        // =====================================================================
        // LOAD FROM CONTIGUOUS LAYOUT
        // =====================================================================
        // k_iter selects thread-pair: ql[2*k_iter], ql[2*k_iter+1]
        // Load both ql values together as int2
        const int2* ql_ptr = reinterpret_cast<const int2*>(blk->ql);
        const int2 ql_pair = ql_ptr[k_iter];
        const int ql_lo = ql_pair.x;
        const int ql_hi = ql_pair.y;
        
        // qh: packed in qh[k_iter] as (qh_lo | qh_hi << 16)
        const uint32_t qh_packed = static_cast<uint32_t>(blk->qh[k_iter]);
        const uint16_t qh_lo = qh_packed & 0xFFFF;        // 1 AND
        const uint16_t qh_hi = qh_packed >> 16;           // 1 SHR
        
        // scale: one per thread-pair
        const half sc = blk->scales[k_iter];
        
        // =====================================================================
        // OPTIMIZED Q6 EXTRACTION - Compute ONLY the k_group-specific pair
        // =====================================================================
        // k_group determines which nibble pair we need:
        //   k_group=0: nib_01 (shift 0)   → q6 elements 0,1
        //   k_group=1: nib_23 (shift 8)   → q6 elements 2,3
        //   k_group=2: nib_45 (shift 4)   → q6 elements 4,5
        //   k_group=3: nib_67 (shift 12)  → q6 elements 6,7
        
        constexpr uint32_t LO_MASK = 0x000f000f;
        
        // =====================================================================
        // OPTIMIZED SHIFT COMPUTATION - Pure bit operations (BRANCHLESS)
        // =====================================================================
        // k_group → shift: 0→0, 1→8, 2→4, 3→12
        // Formula: shift = ((k_group & 1) << 3) | ((k_group & 2) << 1)
        const int nib_shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        // Extract only the nibble pair we need
        const uint32_t nib_lo = (ql_lo >> nib_shift) & LO_MASK;
        const uint32_t nib_hi = (ql_hi >> nib_shift) & LO_MASK;
        
        // Crumb extraction: k_group determines which 2-bit pairs we need
        // For k_group g, we need crumbs at bit positions:
        //   c_even = (qh >> (g*4)) & 0x3      → shifted to bit 4
        //   c_odd  = (qh >> (g*4 + 2)) & 0x3  → shifted to bit 20
        const int crumb_shift = k_group * 4;
        const uint32_t qh_lo_32 = static_cast<uint32_t>(qh_lo);
        const uint32_t qh_hi_32 = static_cast<uint32_t>(qh_hi);
        
        const uint32_t c_lo_even = ((qh_lo_32 >> crumb_shift) & 0x3) << 4;
        const uint32_t c_lo_odd = ((qh_lo_32 >> (crumb_shift + 2)) & 0x3) << 20;
        const uint32_t c_hi_even = ((qh_hi_32 >> crumb_shift) & 0x3) << 4;
        const uint32_t c_hi_odd = ((qh_hi_32 >> (crumb_shift + 2)) & 0x3) << 20;
        
        // Combine nibbles with crumbs to form q6 pairs (already in half-word format)
        const uint32_t p_lo = nib_lo | c_lo_even | c_lo_odd;
        const uint32_t p_hi = nib_hi | c_hi_even | c_hi_odd;
        
        // Single LOP3+HSUB path (no duplication!)
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr int LO_MASK = 0x003f003f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64206420;
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK, EX);
            const half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&w_lo), *reinterpret_cast<const half2*>(&SUB));
            const half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&w_hi), *reinterpret_cast<const half2*>(&SUB));
            
            // Apply scale: w = scale * raw (Q6_K is symmetric, no neg_min)
            const half2 scale2 = __half2half2(sc);
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr int LO_MASK = 0x003f003f;
            constexpr int EX = 0x43004300;
            constexpr uint32_t SUB = 0x43204320;
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK, EX);
            const __nv_bfloat162 raw0 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_lo), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
            const __nv_bfloat162 raw1 = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_hi), *reinterpret_cast<const __nv_bfloat162*>(&SUB));
            
            // Apply scale: w = scale * raw (Q6_K is symmetric, no neg_min)
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc)));
            frag[0] = __hmul2(scale2, raw0);
            frag[1] = __hmul2(scale2, raw1);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_fp8_e4m3>) {
            // FP8 compute type uses FP16 MMA (dequant outputs FP16-packed fragments)
            // Same as half path - keep values in FP16 for MMA compatibility
            constexpr int LO_MASK_Q6 = 0x003f003f;
            constexpr int EX = 0x64006400;
            constexpr uint32_t SUB = 0x64206420;  // 1024 + 32 bias for Q6
            
            int w_lo = lop3<(0xf0 & 0xcc) | 0xaa>(p_lo, LO_MASK_Q6, EX);
            int w_hi = lop3<(0xf0 & 0xcc) | 0xaa>(p_hi, LO_MASK_Q6, EX);
            half2 raw0 = __hsub2(*reinterpret_cast<half2*>(&w_lo), *reinterpret_cast<const half2*>(&SUB));
            half2 raw1 = __hsub2(*reinterpret_cast<half2*>(&w_hi), *reinterpret_cast<const half2*>(&SUB));
            
            // Apply scale in FP16, keep as FP16 for MMA
            const half2 scale2 = __half2half2(sc);
            half2 w0 = __hmul2(scale2, raw0);
            half2 w1 = __hmul2(scale2, raw1);
            frag[0] = *reinterpret_cast<uint32_t*>(&w0);
            frag[1] = *reinterpret_cast<uint32_t*>(&w1);
        }
    }
    
    // =========================================================================
    // DEQUANT FOR 4× MMA m16n8k16 - Half K/128 tile dequant (FULLY OPTIMIZED)
    // =========================================================================
    // Optimizations applied:
    // 1. crumb_shift computed ONCE as k_group << 2
    // 2. qh split into lo/hi halves ONCE at load time (not per-slice)
    // 3. Crumb extraction simplified: (qh >> shift) & 0xF, then reposition
    // 4. Crumbs pre-fused with EX constant: c|EX computed upfront
    // 5. Single LOP3 per element: (nibble & mask) | (crumb|EX) 
    // 6. All type-independent work hoisted before if constexpr
    //
    // Per-slice: shift, LOP3, shift, LOP3, HSUB, HMUL, HSUB, HMUL, 2×store
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
        const uint8_t* row_base = smem_rows + row * K112_BYTES;
        
        // Nibble shift: k_group → 0,8,4,12
        const int shift = ((k_group & 1) << 3) | ((k_group & 2) << 1);
        
        // Compile-time offsets
        constexpr int ql_base = (half_idx == 0) ? 0 : 32;
        constexpr int qh_base = (half_idx == 0) ? 64 : 80;
        constexpr int sc_base = (half_idx == 0) ? 96 : 104;
        
        // Load ql (4 pairs) - 2× LDG.128 instead of 4× LDG.64
        const int4 ql01 = *reinterpret_cast<const int4*>(row_base + ql_base + 0);
        const int4 ql23 = *reinterpret_cast<const int4*>(row_base + ql_base + 16);
        const int2 ql0 = make_int2(ql01.x, ql01.y);
        const int2 ql1 = make_int2(ql01.z, ql01.w);
        const int2 ql2 = make_int2(ql23.x, ql23.y);
        const int2 ql3 = make_int2(ql23.z, ql23.w);
        
        // Load qh - 1× LDG.128 instead of 4× LDG.32
        const int4 qh_all = *reinterpret_cast<const int4*>(row_base + qh_base);
        const uint32_t qh0 = qh_all.x, qh1 = qh_all.y, qh2 = qh_all.z, qh3 = qh_all.w;
        
        // Single lookup gives all extraction params for this k_group
        // Pack: [23:16]=qh_hi_shift, [15:8]=qh_lo_shift, [7:0]=ql_shift (unused, we have 'shift')
        // qh_lo_shift: bits to shift qh to get nibble for .x part (0,4,8,12)
        // qh_hi_shift: bits to shift qh to get nibble for .y part (16,20,24,28)
        static constexpr uint32_t K_PARAMS[4] = {
            (16 << 16) | (0 << 8),   // k_group 0: lo=byte0.lo, hi=byte2.lo
            (20 << 16) | (4 << 8),   // k_group 1: lo=byte0.hi, hi=byte2.hi
            (24 << 16) | (8 << 8),   // k_group 2: lo=byte1.lo, hi=byte3.lo
            (28 << 16) | (12 << 8),  // k_group 3: lo=byte1.hi, hi=byte3.hi
        };
        const uint32_t params = K_PARAMS[k_group];
        const int qh_lo_shift = (params >> 8) & 0xFF;
        const int qh_hi_shift = (params >> 16) & 0xFF;
        
        // Compute crumb|EX inline instead of LUT lookup (eliminates 8× memory dependencies)
        // nibble n → ((n & 3) << 4) | ((n & 0xC) << 18) | EX
        #define CRUMB_EX_FP16(n) (((n & 3) << 4) | ((n & 0xC) << 18) | 0x64006400)
        #define CRUMB_EX_BF16(n) (((n & 3) << 4) | ((n & 0xC) << 18) | 0x43004300)
        
        // Extract crumb nibbles from qh
        const uint32_t n0_lo = (qh0 >> qh_lo_shift) & 0xF;
        const uint32_t n0_hi = (qh0 >> qh_hi_shift) & 0xF;
        const uint32_t n1_lo = (qh1 >> qh_lo_shift) & 0xF;
        const uint32_t n1_hi = (qh1 >> qh_hi_shift) & 0xF;
        const uint32_t n2_lo = (qh2 >> qh_lo_shift) & 0xF;
        const uint32_t n2_hi = (qh2 >> qh_hi_shift) & 0xF;
        const uint32_t n3_lo = (qh3 >> qh_lo_shift) & 0xF;
        const uint32_t n3_hi = (qh3 >> qh_hi_shift) & 0xF;
        
        // Load scales - 1× LDG.64 instead of 4× LDG.16
        const int2 sc_packed = *reinterpret_cast<const int2*>(row_base + sc_base);
        const half sc0 = reinterpret_cast<const half*>(&sc_packed.x)[0];
        const half sc1 = reinterpret_cast<const half*>(&sc_packed.x)[1];
        const half sc2 = reinterpret_cast<const half*>(&sc_packed.y)[0];
        const half sc3 = reinterpret_cast<const half*>(&sc_packed.y)[1];
        
        constexpr uint32_t LO_MASK = 0x000F000F;
        
        // Type-specific dequant using LOP3 + hsub2
        if constexpr (std::is_same_v<compute_t, half>) {
            constexpr uint32_t SUB = 0x64206420;

            // Compute crumb|EX inline (pure ALU, no memory dependency)
            const uint32_t ce0_lo = CRUMB_EX_FP16(n0_lo), ce0_hi = CRUMB_EX_FP16(n0_hi);
            const uint32_t ce1_lo = CRUMB_EX_FP16(n1_lo), ce1_hi = CRUMB_EX_FP16(n1_hi);
            const uint32_t ce2_lo = CRUMB_EX_FP16(n2_lo), ce2_hi = CRUMB_EX_FP16(n2_hi);
            const uint32_t ce3_lo = CRUMB_EX_FP16(n3_lo), ce3_hi = CRUMB_EX_FP16(n3_hi);

            const half2 s0 = __half2half2(sc0), s1 = __half2half2(sc1);
            const half2 s2 = __half2half2(sc2), s3 = __half2half2(sc3);

            // Slice 0: LOP3 does (ql & MASK) | crumb_ex, then subtract and scale
            int w0 = lop3<0xEA>((uint32_t)(ql0.x >> shift), LO_MASK, ce0_lo);
            int w1 = lop3<0xEA>((uint32_t)(ql0.y >> shift), LO_MASK, ce0_hi);
            half2 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            half2 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 1
            w0 = lop3<0xEA>((uint32_t)(ql1.x >> shift), LO_MASK, ce1_lo);
            w1 = lop3<0xEA>((uint32_t)(ql1.y >> shift), LO_MASK, ce1_hi);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 2
            w0 = lop3<0xEA>((uint32_t)(ql2.x >> shift), LO_MASK, ce2_lo);
            w1 = lop3<0xEA>((uint32_t)(ql2.y >> shift), LO_MASK, ce2_hi);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            // Slice 3
            w0 = lop3<0xEA>((uint32_t)(ql3.x >> shift), LO_MASK, ce3_lo);
            w1 = lop3<0xEA>((uint32_t)(ql3.y >> shift), LO_MASK, ce3_hi);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else if constexpr (std::is_same_v<compute_t, __nv_bfloat16>) {
            constexpr uint32_t SUB = 0x43204320;
            
            const uint32_t ce0_lo = CRUMB_EX_BF16(n0_lo), ce0_hi = CRUMB_EX_BF16(n0_hi);
            const uint32_t ce1_lo = CRUMB_EX_BF16(n1_lo), ce1_hi = CRUMB_EX_BF16(n1_hi);
            const uint32_t ce2_lo = CRUMB_EX_BF16(n2_lo), ce2_hi = CRUMB_EX_BF16(n2_hi);
            const uint32_t ce3_lo = CRUMB_EX_BF16(n3_lo), ce3_hi = CRUMB_EX_BF16(n3_hi);
            
            const __nv_bfloat162 s0 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc0)));
            const __nv_bfloat162 s1 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc1)));
            const __nv_bfloat162 s2 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc2)));
            const __nv_bfloat162 s3 = __bfloat162bfloat162(__float2bfloat16(__half2float(sc3)));
            
            int w0 = lop3<0xEA>((uint32_t)(ql0.x >> shift), LO_MASK, ce0_lo);
            int w1 = lop3<0xEA>((uint32_t)(ql0.y >> shift), LO_MASK, ce0_hi);
            __nv_bfloat162 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            __nv_bfloat162 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql1.x >> shift), LO_MASK, ce1_lo);
            w1 = lop3<0xEA>((uint32_t)(ql1.y >> shift), LO_MASK, ce1_hi);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql2.x >> shift), LO_MASK, ce2_lo);
            w1 = lop3<0xEA>((uint32_t)(ql2.y >> shift), LO_MASK, ce2_hi);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql3.x >> shift), LO_MASK, ce3_lo);
            w1 = lop3<0xEA>((uint32_t)(ql3.y >> shift), LO_MASK, ce3_hi);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w0), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w1), *reinterpret_cast<const __nv_bfloat162*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
            
        } else {
            // FP8: use FP16 path
            constexpr uint32_t SUB = 0x64206420;
            
            const uint32_t ce0_lo = CRUMB_EX_FP16(n0_lo), ce0_hi = CRUMB_EX_FP16(n0_hi);
            const uint32_t ce1_lo = CRUMB_EX_FP16(n1_lo), ce1_hi = CRUMB_EX_FP16(n1_hi);
            const uint32_t ce2_lo = CRUMB_EX_FP16(n2_lo), ce2_hi = CRUMB_EX_FP16(n2_hi);
            const uint32_t ce3_lo = CRUMB_EX_FP16(n3_lo), ce3_hi = CRUMB_EX_FP16(n3_hi);
            
            const half2 s0 = __half2half2(sc0), s1 = __half2half2(sc1);
            const half2 s2 = __half2half2(sc2), s3 = __half2half2(sc3);
            
            int w0 = lop3<0xEA>((uint32_t)(ql0.x >> shift), LO_MASK, ce0_lo);
            int w1 = lop3<0xEA>((uint32_t)(ql0.y >> shift), LO_MASK, ce0_hi);
            half2 r0 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            half2 r1 = __hmul2(s0, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[0] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[1] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql1.x >> shift), LO_MASK, ce1_lo);
            w1 = lop3<0xEA>((uint32_t)(ql1.y >> shift), LO_MASK, ce1_hi);
            r0 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s1, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[2] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[3] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql2.x >> shift), LO_MASK, ce2_lo);
            w1 = lop3<0xEA>((uint32_t)(ql2.y >> shift), LO_MASK, ce2_hi);
            r0 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s2, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[4] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[5] = *reinterpret_cast<uint32_t*>(&r1);
            
            w0 = lop3<0xEA>((uint32_t)(ql3.x >> shift), LO_MASK, ce3_lo);
            w1 = lop3<0xEA>((uint32_t)(ql3.y >> shift), LO_MASK, ce3_hi);
            r0 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w0), *reinterpret_cast<const half2*>(&SUB)));
            r1 = __hmul2(s3, __hsub2(*reinterpret_cast<half2*>(&w1), *reinterpret_cast<const half2*>(&SUB)));
            frag_b[6] = *reinterpret_cast<uint32_t*>(&r0);
            frag_b[7] = *reinterpret_cast<uint32_t*>(&r1);
        }
        
        #undef CRUMB_EX_FP16
        #undef CRUMB_EX_BF16
    }
};

// Q6_KO: Q6_K's compact block is already in ordered form (ql contiguous, qh
// contiguous, scales at the tail), so the KO twin inherits the entire Q6_K trait and
// only overrides the int8 ql load to exploit the contiguous layout — the sub's 4 ql
// ints load in ONE int4 (broadcast across the k-group) instead of two int loads. The
// qh-crumb spread + center math is identical to Q6_K. Byte-identical result.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q6_KO, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q6_K, compute_t, scale_t> {
    using base = gemx_dequant_traits<block_c_q6_K, compute_t, scale_t>;

    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        const int row = lane >> 2;
        const int q3 = lane & 3;
        // De-interleaved Q6_KO block is 96B (quant only); scales are in the separate region.
        const uint8_t* rb = warp_rows + row * smem_row_stride<block_c_q6_KO_k128>::value;
        const int sh = (q3 & 1) * 4;
        const int m0 = sub * 4 + (q3 >> 1);
        const int m1 = m0 + 2;
        // int4 over the sub's 4 ql ints (contiguous at sub*16); pick m0/m1 from regs.
        const int4 v = *reinterpret_cast<const int4*>(rb + sub * 16);
        const int v0 = (q3 < 2) ? v.x : v.y;
        const int v1 = (q3 < 2) ? v.z : v.w;
        const uint32_t nib0 = __byte_perm((v0 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t nib1 = __byte_perm((v1 >> sh) & 0x0F0F0F0F, 0, 0x3120);
        const uint32_t qh0 = *reinterpret_cast<const uint16_t*>(rb + 64 + (m0 >> 1) * 4 + (m0 & 1) * 2);
        const uint32_t qh1 = *reinterpret_cast<const uint16_t*>(rb + 64 + (m1 >> 1) * 4 + (m1 & 1) * 2);
        const uint32_t cr0 = (qh0 >> ((q3 & 1) * 8)) & 0xFFu;
        const uint32_t cr1 = (qh1 >> ((q3 & 1) * 8)) & 0xFFu;
        // Per-32 AFFINE Q6_KO: the 6-bit value stays UNSIGNED (0..63, fits int8 positive) —
        // the per-32 (scale,min) fold handles the offset. No ^0x20 centering / sign-extend.
        b_frag[0] = nib0 | ((cr0 & 0x03u) << 4) | ((cr0 & 0x0Cu) << 10)
                        | ((cr0 & 0x30u) << 16) | ((cr0 & 0xC0u) << 22);
        b_frag[1] = nib1 | ((cr1 & 0x03u) << 4) | ((cr1 & 0x0Cu) << 10)
                        | ((cr1 & 0x30u) << 16) | ((cr1 & 0xC0u) << 22);
    }
};

// Q6_KO K/1024 chunk — WAVEFRONT-OPTIMAL dequant, LANE-MAJOR. The 768 B quant region splits
// into a 512 B ql stream (lane's 4 subs at lane*16+sub*4, like Q4) and a 256 B qh crumb stream
// (lane's 4 subs' uint16s at 512+lane*8+sub*2). ONE int4 LDS at lane*16 pulls all 4 subs' ql,
// ONE int2 LDS at 512+lane*8 pulls all 4 subs' crumb-uint16s — both conflict-free. Per sub: low
// nibbles = b_frag[0] low4 (K[q3*4..+3]), high nibbles = b_frag[1] low4 (K[q3*4+16..+19]); the
// uint16's cr0/cr1 spread into bits 4-5 of each output byte (Q6_K math). Values UNSIGNED
// (0..63); the per-32 (scale,min) fold handles the offset. dm at 768.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q6_KO_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q6_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int4 vv = *reinterpret_cast<const int4*>(chunk + lane * 16);
        const uint32_t s4[4] = {(uint32_t)vv.x, (uint32_t)vv.y, (uint32_t)vv.z, (uint32_t)vv.w};
        const int2 cc = *reinterpret_cast<const int2*>(chunk + 512 + lane * 8);
        const uint32_t c2[2] = {(uint32_t)cc.x, (uint32_t)cc.y};   // c2[0]: subs 0,1  c2[1]: subs 2,3
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t nib0 = s4[sub] & 0x0F0F0F0Fu;
            const uint32_t nib1 = (s4[sub] >> 4) & 0x0F0F0F0Fu;
            const uint32_t qh16 = (c2[sub >> 1] >> ((sub & 1) * 16)) & 0xFFFFu;
            const uint32_t cr0 = qh16 & 0xFFu;
            const uint32_t cr1 = (qh16 >> 8) & 0xFFu;
            b_frags[sub][0] = nib0 | ((cr0 & 0x03u) << 4) | ((cr0 & 0x0Cu) << 10)
                                  | ((cr0 & 0x30u) << 16) | ((cr0 & 0xC0u) << 22);
            b_frags[sub][1] = nib1 | ((cr1 & 0x03u) << 4) | ((cr1 & 0x0Cu) << 10)
                                  | ((cr1 & 0x30u) << 16) | ((cr1 & 0xC0u) << 22);
        }
    }
};

// Simplified type aliases
using Q6K_Dequant_FP16 = gemx_dequant_traits<block_c_q6_K, half, half>;
using Q6K_Dequant_BF16 = gemx_dequant_traits<block_c_q6_K, __nv_bfloat16, __nv_bfloat16>;
using Q6K_Dequant_FP8 = gemx_dequant_traits<block_c_q6_K, __nv_fp8_e4m3, half>;
