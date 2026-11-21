// The K-tile loader for Q5_K and the scale extraction it uses for repacking.
//
// `block_c_q5_K_ktile` is the K-tile loader's block type, and
// `vec_dot_q_loader_q5_K_inline` fills the K tiles. The `vec_dot_loader_for`
// specializations cover the compact block types, and the `gemx_q5_K` namespace
// extracts scales for repacking.

// GEMX permutation helpers (unused in K-tile-major path)
__device__ __forceinline__ int q5k_elem_from_ql_dst(int ql_dst) {
    return (ql_dst >> 6) + (((ql_dst >> 1) & 31) << 2) + ((ql_dst & 1) << 7);
}

__device__ __forceinline__ int q5k_qh_dst_from_ql_dst(int ql_dst) {
    return (((ql_dst & 127) >> 1) << 2) + ((ql_dst >> 7) << 1) + (ql_dst & 1);
}

// Decodes scale from inline block structure (unused in K-tile-major path)
template <typename acc_t>
__device__ __forceinline__ acc_t decode_q5k_scale_inline(
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
__device__ __forceinline__ acc_t load_q5k_external_scale(
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
// K-TILE LOADER FOR Q5_K
// =============================================================================
// Uses block_c_q5_K_ktile which has the same 12-byte layout as block_c_q6_K_ktile:
//   - ql_x: 4B (nibbles 0-7)
//   - ql_y: 4B (nibbles 8-15)
//   - qh:   4B (16 × 1-bit high bits, upper 16 bits padded with zeros)

// Local K-tile struct for loader storage (16 elements)
typedef struct __align__(4) {
    int ql_x;
    int ql_y;
    int qh;
} block_c_q5_K_ktile;

template <int vdr, typename acc_t>
struct vec_dot_q_loader_q5_K_inline {
    // TYPE ALIASES
    using acc2_type = acc2_for_act_t<acc_t>;
    
    // Option E: 8 elements per thread (single int of nibbles + 8 high bits)
    int ql;       // 8 nibbles (4 bits each)
    uint32_t qh8; // 8 high bits (1 bit each)
    acc2_type sm; // (scale, neg_min) in native format for acc_t
    
    // Option E: 16 threads per K/128 block
    __device__ __forceinline__ static int get_lane() {
        return threadIdx.x & 15;  // 16 lanes for K/128
    }

    // 8-PART LOAD INTERFACE (8 elements per part) - Option E
    template <int N>
    __device__ __forceinline__ void load_part(
        const block_c_q5_K* __restrict__ x,
        int row,
        int kbx,
        int num_rows
    ) {
        static_assert(N < 16, "Q5_K uses 16-part interface (K/128)");

        const int block_idx = kbx * num_rows + row;
        const block_c_q5_K_k128* __restrict__ blk = reinterpret_cast<const block_c_q5_K_k128*>(&x[block_idx]);

        // K/128 layout via data[] array
        // data[0]=dm0, data[1]=qh0123, data[2-5]=qs0-3, data[6-9]=qs4-7,
        // data[10]=qh4567, data[11]=dm1, data[12]=dm2, data[13]=qh891011,
        // data[14-17]=qs8-11, data[18-21]=qs12-15, data[22]=qh12131415, data[23]=dm3
        const int lane = get_lane();
        
        // qs indices mapping
        static constexpr int qs_idx[16] = {2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21};
        ql = blk->data[qs_idx[lane]];
        
        // High bits: qh0123=data[1], qh4567=data[10], qh891011=data[13], qh12131415=data[22]
        static constexpr int qh_idx[4] = {1, 10, 13, 22};
        const uint32_t qh_word = static_cast<uint32_t>(blk->data[qh_idx[lane >> 2]]);
        const int shift = (lane & 3) * 8;
        qh8 = (qh_word >> shift) & 0xFF;
        
        // Scales: dm0=data[0], dm1=data[11], dm2=data[12], dm3=data[23]
        static constexpr int dm_idx[4] = {0, 11, 12, 23};
        const half2* dm_ptr = reinterpret_cast<const half2*>(&blk->data[dm_idx[lane >> 2]]);
        sm = convert_half2_to_acc2<acc2_type>(*dm_ptr);
    }
    
    // DOT PRODUCT - Option E: 8 elements per thread
    // Q5_K: 5-bit = 4-bit nibble + 1-bit high bit, affine quantization
    //
    // LOP3-READY Memory layout (from repacker pack_nibbles_lop3_ready):
    //   ql bits: [3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6,
    //            [19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
    //   qh8 uint8: h0h1h2h3h4h5h6h7 (each hi is 1 bit at position i)
    //
    // LOP3-READY Nibble extraction (with LO_MASK = 0x000f000f):
    //   ql & LO_MASK           → (n0, n1) pair
    //   (ql >> 8) & LO_MASK    → (n2, n3) pair
    //   (ql >> 4) & LO_MASK    → (n4, n5) pair
    //   (ql >> 12) & LO_MASK   → (n6, n7) pair
    //
    // We need: sum_i(q5[i] * y[i]) where q5[i] = (nibble[i] | (high_bit[i] << 4))
    // Then apply affine: scale * q5 + neg_min
    template <int N, typename acc_t_inner, typename y_t>
    __device__ __forceinline__ acc_t_inner dot_y(const y_t* __restrict__ y) const {
        static_assert(N < 16, "Q5_K uses 16-part interface (K/128)");
        
        // LOP3-READY: Extract nibble pairs directly using shifts
        constexpr uint32_t LO_MASK = 0x000f000f;
        const uint32_t nib_01 = ql & LO_MASK;               // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;        // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;        // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;       // (n6, n7)
        
        // Extract high bits from qh8 and build pairs matching nibble pairs
        // qh8 layout: h0(bit 0), h1(bit 1), ..., h7(bit 7)
        // Build high bit pairs shifted left by 4 to OR with nibbles
        const uint32_t hb_01 = ((qh8 & 0x1) << 4) | (((qh8 >> 1) & 0x1) << 20);
        const uint32_t hb_23 = (((qh8 >> 2) & 0x1) << 4) | (((qh8 >> 3) & 0x1) << 20);
        const uint32_t hb_45 = (((qh8 >> 4) & 0x1) << 4) | (((qh8 >> 5) & 0x1) << 20);
        const uint32_t hb_67 = (((qh8 >> 6) & 0x1) << 4) | (((qh8 >> 7) & 0x1) << 20);
        
        // Combine nibbles and high bits to form q5 pairs
        const uint32_t q5_01 = nib_01 | hb_01;  // (q5[0], q5[1])
        const uint32_t q5_23 = nib_23 | hb_23;  // (q5[2], q5[3])
        const uint32_t q5_45 = nib_45 | hb_45;  // (q5[4], q5[5])
        const uint32_t q5_67 = nib_67 | hb_67;  // (q5[6], q5[7])
        
        // Extract individual q5 values from pairs (5-bit values in half-word positions)
        const int q0 = int(q5_01 & 0x1F);
        const int q1 = int((q5_01 >> 16) & 0x1F);
        const int q2 = int(q5_23 & 0x1F);
        const int q3 = int((q5_23 >> 16) & 0x1F);
        const int q4 = int(q5_45 & 0x1F);
        const int q5 = int((q5_45 >> 16) & 0x1F);
        const int q6 = int(q5_67 & 0x1F);
        const int q7 = int((q5_67 >> 16) & 0x1F);
        
        if constexpr (std::is_same_v<y_t, float>) {
            const float scale = to_f32(lo_acc2(sm));
            const float neg_min = to_f32(hi_acc2(sm));
            const float4* y4 = reinterpret_cast<const float4*>(y + get_lane() * 8);
            
            float sum = 0.0f;
            {
                const float4 yv = y4[0];
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q0), neg_min), yv.x, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q1), neg_min), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q2), neg_min), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q3), neg_min), yv.w, sum);
            }
            {
                const float4 yv = y4[1];
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q4), neg_min), yv.x, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q5), neg_min), yv.y, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q6), neg_min), yv.z, sum);
                sum  = __fmaf_rn(__fmaf_rn(scale, float(q7), neg_min), yv.w, sum);
            }
            return sum;
            
        } else if constexpr (std::is_same_v<y_t, half>) {
            // HALF PATH with LOP3 + PRMT OPTIMIZATION
            // Q5_K: 5-bit values (0-31), use LOP3 magic number conversion
            const half2 scale2 = __half2half2(lo_acc2(sm));
            const half2 neg_min2 = __half2half2(hi_acc2(sm));
            const half2* y2 = reinterpret_cast<const half2*>(y + get_lane() * 8);
            
            // LOP3 magic constants for FP16:
            // EX = 0x64006400 is exp=10 (bias 15), so mantissa bits become value+1024
            // BIAS = 0x64006400 = half2(1024, 1024)
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;  // half2(1024, 1024)
            constexpr int LO_MASK_5BIT = 0x001f001f;  // 5-bit mask
            
            // LOP3-READY: Direct conversion without PRMT
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[3], sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
            
        } else if constexpr (std::is_same_v<y_t, __nv_bfloat16>) {
            // BF16 PATH with LOP3 + PRMT OPTIMIZATION
            // Q5_K: 5-bit values (0-31), use LOP3 magic number conversion
            const __nv_bfloat162 scale2 = __bfloat162bfloat162(lo_acc2(sm));
            const __nv_bfloat162 neg_min2 = __bfloat162bfloat162(hi_acc2(sm));
            const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y + get_lane() * 8);
            
            // LOP3 magic constants for BF16:
            constexpr int EX_BF = 0x43004300;
            constexpr uint32_t BIAS_BF = 0x43004300;  // bf162(128, 128)
            constexpr int LO_MASK_5BIT = 0x001f001f;  // 5-bit mask
            
            // LOP3-READY: Direct conversion without PRMT
            __nv_bfloat162 sum2 = __float2bfloat162_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[0], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[1], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX_BF);
                __nv_bfloat162 w = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&w_raw),
                                           *reinterpret_cast<const __nv_bfloat162*>(&BIAS_BF));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2[2], sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX_BF);
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
            static_assert(sizeof(y_t) == 1, "Unexpected type in dot_y - expected FP8 (1 byte)");
            
            const half scale_h = lo_acc2(sm);
            const half neg_min_h = hi_acc2(sm);
            const half2 scale2 = __half2half2(scale_h);
            const half2 neg_min2 = __half2half2(neg_min_h);
            
            // Load FP8 inputs as uint32_t (4 FP8 values per load = 8 values in 2 loads)
            const uint32_t* y_u32 = reinterpret_cast<const uint32_t*>(y + get_lane() * 8);
            const uint32_t y_packed0 = y_u32[0];  // y[0..3] as FP8
            const uint32_t y_packed1 = y_u32[1];  // y[4..7] as FP8
            
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
            
            // LOP3-READY: Direct conversion without PRMT
            constexpr int EX = 0x64006400;
            constexpr uint32_t BIAS = 0x64006400;
            constexpr int LO_MASK_5BIT = 0x001f001f;  // 5-bit mask
            
            half2 sum2 = __float2half2_rn(0.0f);
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_01, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y0, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_23, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y1, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_45, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y2_v, sum2);
            }
            {
                int w_raw = lop3<(0xf0 & 0xcc) | 0xaa>(q5_67, LO_MASK_5BIT, EX);
                half2 w = __hsub2(*reinterpret_cast<half2*>(&w_raw), *reinterpret_cast<const half2*>(&BIAS));
                sum2 = __hfma2(__hfma2(scale2, w, neg_min2), y3, sum2);
            }
            
            return __half2float(__low2half(sum2)) + __half2float(__high2half(sum2));
        }
    }
    
    // DEQUANTIZE - 8 elements per thread = 2 float4 stores
    template <int N>
    __device__ __forceinline__ void dequant(
        float* __restrict__ out
    ) const {
        static_assert(N < 16, "Q5_K uses 16-part interface (K/128)");
        
        const float scale_f = to_f32(lo_acc2(sm));
        const float neg_min = to_f32(hi_acc2(sm));
        
        // LOP3-READY: Extract nibble pairs using shift-based extraction
        constexpr uint32_t LO_MASK = 0x000f000f;
        const uint32_t nib_01 = ql & LO_MASK;               // (n0, n1)
        const uint32_t nib_23 = (ql >> 8) & LO_MASK;        // (n2, n3)
        const uint32_t nib_45 = (ql >> 4) & LO_MASK;        // (n4, n5)
        const uint32_t nib_67 = (ql >> 12) & LO_MASK;       // (n6, n7)
        
        // Extract high bits from qh8 and build pairs matching nibble pairs
        // qh8 layout: h0(bit 0), h1(bit 1), ..., h7(bit 7)
        // Build high bit pairs shifted left by 4 to OR with nibbles
        const uint32_t hb_01 = ((qh8 & 0x1) << 4) | (((qh8 >> 1) & 0x1) << 20);
        const uint32_t hb_23 = (((qh8 >> 2) & 0x1) << 4) | (((qh8 >> 3) & 0x1) << 20);
        const uint32_t hb_45 = (((qh8 >> 4) & 0x1) << 4) | (((qh8 >> 5) & 0x1) << 20);
        const uint32_t hb_67 = (((qh8 >> 6) & 0x1) << 4) | (((qh8 >> 7) & 0x1) << 20);
        
        // Combine nibbles and high bits to form q5 pairs
        const uint32_t q5_01 = nib_01 | hb_01;  // (q5[0], q5[1])
        const uint32_t q5_23 = nib_23 | hb_23;  // (q5[2], q5[3])
        const uint32_t q5_45 = nib_45 | hb_45;  // (q5[4], q5[5])
        const uint32_t q5_67 = nib_67 | hb_67;  // (q5[6], q5[7])
        
        // Extract individual q5 values and apply affine transform
        const float q0_f = __fmaf_rn(scale_f, float(q5_01 & 0x1F), neg_min);
        const float q1_f = __fmaf_rn(scale_f, float((q5_01 >> 16) & 0x1F), neg_min);
        const float q2_f = __fmaf_rn(scale_f, float(q5_23 & 0x1F), neg_min);
        const float q3_f = __fmaf_rn(scale_f, float((q5_23 >> 16) & 0x1F), neg_min);
        const float q4_f = __fmaf_rn(scale_f, float(q5_45 & 0x1F), neg_min);
        const float q5_f = __fmaf_rn(scale_f, float((q5_45 >> 16) & 0x1F), neg_min);
        const float q6_f = __fmaf_rn(scale_f, float(q5_67 & 0x1F), neg_min);
        const float q7_f = __fmaf_rn(scale_f, float((q5_67 >> 16) & 0x1F), neg_min);
        
        float4* out4 = reinterpret_cast<float4*>(out + get_lane() * 8);
        
        // Output in correct order: q0,q1,q2,q3, q4,q5,q6,q7
        out4[0] = make_float4(q0_f, q1_f, q2_f, q3_f);
        out4[1] = make_float4(q4_f, q5_f, q6_f, q7_f);
    }
};

// LOADER TRAIT SPECIALIZATIONS
template <typename act_t>
struct vec_dot_loader_for<block_q5_K, 1, act_t> {
    using type = vec_dot_q_loader_q5_K_inline<1, float>;
};

template <typename act_t>
struct vec_dot_loader_for<block_q5_K, 2, act_t> {
    using type = vec_dot_q_loader_q5_K_inline<2, float>;
};

// K/128 compact block type specializations
template <typename act_t>
struct vec_dot_loader_for<block_c_q5_K, 1, act_t> {
    using type = vec_dot_q_loader_q5_K_inline<1, float>;
};

template <typename act_t>
struct vec_dot_loader_for<block_c_q5_K, 2, act_t> {
    using type = vec_dot_q_loader_q5_K_inline<2, float>;
};

// =============================================================================
// SCALE EXTRACTION FOR REPACKING
// =============================================================================
// Q5_K uses the same scale encoding as Q4_K:
// - 8 sub-blocks of 32 elements each per 256-element super-block
// - Each sub-block has a 6-bit scale and 6-bit min packed into 12 bytes
// - Output: half2 (d*scale_6bit, -dmin*min_6bit) per 32 elements

namespace gemx_q5_K {

struct Q5K_Traits {
    static constexpr int BLOCK_ELEMENTS = 256;
    static constexpr int INPUT_BYTES = 176;   // Original Q5_K block size
    static constexpr int OUTPUT_BYTES = 192;  // 16 K-tiles × 12B each
    static constexpr int QL_BYTES = 128;      // 16 × 8B low nibbles
    static constexpr int QH_BYTES = 64;       // 16 × 4B high bits (padded from 2B)
    static constexpr bool NEEDS_PERMUTATION = true;
    static constexpr int THREADS_PER_BLOCK = 32;
};

// Extract single scale+min pair: (d*scale_6bit, -dmin*min_6bit) → half2
// Same encoding as Q4_K - 6-bit scale/min packed into 12 bytes
template <typename ScaleT>
__device__ __forceinline__ void extract_scale(
    const block_q5_K* __restrict__ block,
    ScaleT* __restrict__ scales_out,
    int dst_scale_idx,
    int sub_idx
) {
    float2 dm = __half22float2(block->dm);
    float d = dm.x;
    float dmin = dm.y;
    
    const uint8_t* sc = block->scales;
    
    // Q5_K uses same 6-bit scale/min packing as Q4_K
    int scale_6bit, min_6bit;
    if (sub_idx < 4) {
        scale_6bit = sc[sub_idx] & 0x3F;
        min_6bit = sc[sub_idx + 4] & 0x3F;
    } else {
        scale_6bit = ((sc[sub_idx + 4] & 0x0F) | ((sc[sub_idx - 4] >> 6) << 4));
        min_6bit = ((sc[sub_idx + 4] >> 4) | ((sc[sub_idx] >> 6) << 4));
    }
    
    float effective_scale = d * float(scale_6bit);
    float effective_neg_min = -dmin * float(min_6bit);
    
    scales_out[dst_scale_idx] = make_scale_pair<ScaleT>(effective_scale, effective_neg_min);
}

// Extract all scales: row-major [N, K/256] → column-major [K/32, N]
// 8 scales per super-block, each covering 32 elements (2 K-tiles)
template <typename ScaleT>
__device__ inline void extract_scales_impl(
    const block_q5_K* __restrict__ x,
    ScaleT* __restrict__ scales_out,
    int nrows,
    int ncols
) {
    constexpr int SCALES_PER_SUPERBLOCK = 8;  // 8 sub-blocks of 32 elements
    constexpr int ELEMENTS_PER_SUPERBLOCK = 256;
    constexpr int ELEMENTS_PER_SCALE = 32;
    const int superblocks_per_row = ncols / ELEMENTS_PER_SUPERBLOCK;
    const int scales_per_row = ncols / ELEMENTS_PER_SCALE;  // K/32 scales per row
    const int total_scales = nrows * scales_per_row;
    
    for (int src_scale_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         src_scale_idx < total_scales; 
         src_scale_idx += blockDim.x * gridDim.x) 
    {
        // Source layout: row-major [nrows, scales_per_row]
        const int row = src_scale_idx / scales_per_row;
        const int scale_col = src_scale_idx % scales_per_row;
        
        // Which super-block and sub-block within it
        const int superblock_col = scale_col / SCALES_PER_SUPERBLOCK;
        const int sub_idx = scale_col % SCALES_PER_SUPERBLOCK;
        const int superblock_idx = row * superblocks_per_row + superblock_col;
        
        // Destination layout: simple column-major [K/32, N] - NO GEMX permutation
        const int dst_scale_idx = scale_col * nrows + row;
        
        extract_scale<ScaleT>(&x[superblock_idx], scales_out, dst_scale_idx, sub_idx);
    }
}

} // namespace gemx_q5_K
