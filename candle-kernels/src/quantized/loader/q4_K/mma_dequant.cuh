// The Q4_K int8 weight unpack and the GEMX dequant traits for Q4_K and its
// byte-permuted Q4_KO form.
//
// `q4k_dequant_to_b_frag_int8` and `q4k_sub_dm` are the shared Q4_K and Q4_KO
// int8 unpack, since the two formats hold identical qs nibbles.
// `gemx_dequant_traits` covers Q4_K and Q4_KO, whose K/1024 chunk is the
// wavefront-optimal int8 dequant in lane-major order.

// Shared Q4_K / Q4_KO int8 weight unpack. The two formats hold identical qs nibbles
// and (scale,-min) pairs; they differ ONLY in where a sub's 4 qs ints and its scale
// pair sit inside the 80-byte block (Q4_KO groups qs contiguously and the scales at
// the tail). Both traits delegate here with their own offset tables — the nibble
// gather (mask + prmt 0x3120) is byte-identical. `sub` is a compile-time unroll
// index at the call site, so qs_base[sub]/dm_off[sub] fold to constants.
__device__ __forceinline__ void q4k_dequant_to_b_frag_int8(
    const uint8_t* __restrict__ warp_rows, int sub, int lane,
    uint32_t (&b_frag)[2], const int (&qs_base)[4], int block_bytes)
{
    const int row = lane >> 2;
    const int q3  = lane & 3;
    const uint8_t* rb = warp_rows + row * block_bytes;
    const int qlo = *reinterpret_cast<const int*>(rb + qs_base[sub] + (q3 >> 1) * 4);
    const int qhi = *reinterpret_cast<const int*>(rb + qs_base[sub] + (2 + (q3 >> 1)) * 4);
    const int sh = (q3 & 1) * 4;  // 0 = low nibbles (n0..n3), 4 = high (n4..n7)
    b_frag[0] = __byte_perm((qlo >> sh) & 0x0F0F0F0F, 0, 0x3120);
    b_frag[1] = __byte_perm((qhi >> sh) & 0x0F0F0F0F, 0, 0x3120);
}

__device__ __forceinline__ half2 q4k_sub_dm(
    const uint8_t* __restrict__ row_block, int sub, const int (&dm_off)[4])
{
    return *reinterpret_cast<const half2*>(row_block + dm_off[sub]);
}

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

    // -------------------------------------------------------------------------
    // INT8 PATH (grouped_tc_int8): register-direct n8k32 B-fragment, no smem
    // staging. lane provides output-row n=(lane>>2)'s Q4_K weights for sub-block
    // `sub`, packed exactly in the load_b_frag_n8k32 layout:
    //   b_frag[0] = 4 int8 at K_local (lane&3)*4 + {0..3}
    //   b_frag[1] = 4 int8 at K_local (lane&3)*4 + 16 + {0..3}
    // Unsigned nibbles (0..15); the fold applies w = d·nibble + m as d·C + m·Σâ,
    // reading d/m via sub_dm() below — i.e. the FP loader's
    // __hfma2(scale, nibble, neg_min) with the multiply moved past the MMA.
    // Nibble bit-positions n0..n7 match extract_4_elements.
    // -------------------------------------------------------------------------
    // Q4_K layout: each sub's 4 qs ints are split around the interspersed scales
    // ({0,24,40,64}); the four (scale,-min) pairs live at {16,20,56,60}.
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows,  // this warp's 8 raw Q4_K rows
        int sub, int lane, uint32_t (&b_frag)[2])
    {
        constexpr int QS_BASE[4] = {0, 24, 40, 64};  // byte offset of the sub's 4 qs ints
        q4k_dequant_to_b_frag_int8(warp_rows, sub, lane, b_frag, QS_BASE, K128_BYTES);
    }

    // Per-sub {scale d (low), neg_min m (high)} for one weight row's raw block.
    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {16, 20, 56, 60};
        return q4k_sub_dm(row_block, sub, DM_OFF);
    }
    
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

// =============================================================================
// GEMX DEQUANT TRAITS - Q4_KO (byte-permuted Q4_K for the q8a128 int8 path)
// =============================================================================
// Q4_KO holds the same nibbles and scales as Q4_K, only reordered: qs contiguous
// (per-sub at {0,16,32,48}), the four (scale,-min) pairs grouped at the tail
// ({64,68,72,76}). It inherits the entire Q4_K trait and overrides ONLY the two
// int8 accessors with the permuted offsets — the int8 m16n8k32 path is the only
// consumer (no FP KO kernel is instantiated). The inherited FP accessors still
// assume the Q4_K byte order, so a KO FP kernel would need them overridden too.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_K, compute_t, scale_t> {
    using base = gemx_dequant_traits<block_c_q4_K, compute_t, scale_t>;

    // Q4_KO interleaves each sub's 4 qs ints as [I0,I2,I1,I3] so the lane's qlo (the
    // K[0:16] int) and qhi (the K[16:32] int) sit ADJACENT: one int2 (LDS.64) then
    // loads exactly the 8 bytes the lane needs — halving the qs LSU instructions vs
    // two int loads, with none of the int4 bandwidth waste (which read 16B to use 8).
    //   q3<2 → int2 at sub*16+0 = {I0,I2} = {qlo,qhi};  q3>=2 → sub*16+8 = {I1,I3}.
    __device__ __forceinline__ static void dequant_to_b_frag_int8(
        const uint8_t* __restrict__ warp_rows, int sub, int lane, uint32_t (&b_frag)[2])
    {
        const int row = lane >> 2;
        const int q3 = lane & 3;
        // De-interleaved Q4_KO block is 64B (quant only); scales are in the separate region.
        const uint8_t* rb = warp_rows + row * smem_row_stride<block_c_q4_KO_k128>::value;
        const int2 qv = *reinterpret_cast<const int2*>(rb + sub * 16 + (q3 >> 1) * 8);
        const int sh = (q3 & 1) * 4;
        b_frag[0] = __byte_perm((qv.x >> sh) & 0x0F0F0F0F, 0, 0x3120);
        b_frag[1] = __byte_perm((qv.y >> sh) & 0x0F0F0F0F, 0, 0x3120);
    }

    __device__ __forceinline__ static half2 sub_dm(const uint8_t* __restrict__ row_block, int sub) {
        constexpr int DM_OFF[4] = {64, 68, 72, 76};  // scales grouped at the tail
        return q4k_sub_dm(row_block, sub, DM_OFF);
    }
};

// Q4_KO K/1024 chunk — WAVEFRONT-OPTIMAL ("perfect") int8 dequant, LANE-MAJOR. The 512 B quant
// region stores each lane's 4 sub-uint32s contiguously: byte for (lane, sub, i) at lane*16 +
// sub*4 + i, where the uint32 still packs nibbles as K[p] | (K[p+16]<<4). So ONE int4 LDS at
// lane*16 pulls all 4 subs (vs 4 separate uint32 LDS) — a quarter of the MIO instructions.
// Bank-conflict-free: an LDS.128 services the warp in 4 quarter-warp phases (8 lanes × 16 B =
// 128 B = 32 banks); lane L touches banks {4L..4L+3}, and each phase covers banks 0-31 exactly
// once. Per sub, low nibbles ARE b_frag[0] (K[q3*4..+3]) and high nibbles ARE b_frag[1]
// (K[q3*4+16..+19]) — no byte_perm. The inline scales blk.dm[row] are read in the kernel.
template <typename compute_t, typename scale_t>
struct gemx_dequant_traits<block_c_q4_KO_k1024, compute_t, scale_t>
    : gemx_dequant_traits<block_c_q4_KO, compute_t, scale_t> {
    __device__ __forceinline__ static void dequant_all_subs_int8(
        const uint8_t* __restrict__ chunk, int lane, uint32_t (&b_frags)[4][2])
    {
        const int4 vv = *reinterpret_cast<const int4*>(chunk + lane * 16);
        const uint32_t s4[4] = {(uint32_t)vv.x, (uint32_t)vv.y, (uint32_t)vv.z, (uint32_t)vv.w};
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            b_frags[sub][0] = s4[sub] & 0x0F0F0F0Fu;          // low nibbles  → K[q3*4 .. +3]
            b_frags[sub][1] = (s4[sub] >> 4) & 0x0F0F0F0Fu;   // high nibbles → K[q3*4+16 .. +19]
        }
    }
};
