// The input block structs and the repacks of the simple formats and Q8_K.
//
// Declares the `block_q*_t` input structs the repacks read and the unaligned
// load helpers, then the repack functions `repack_q4_0_impl`,
// `repack_q4_1_impl`, `repack_q5_0_impl`, `repack_q5_1_impl`,
// `repack_q8_0_impl`, `repack_q8_1_impl` and `repack_q8_K_impl`, which write
// K/128 output with the embedded scales.

// =============================================================================
// QUANTIZED BLOCK STRUCTURES (for input format understanding)
// =============================================================================
// These match the GGML block formats used in quantized tensors.
// We need these to know input layout for repacking.

typedef struct {
    half    d;              // delta (scale)
    uint8_t qs[QK4_0 / 2];  // nibbles / quants (16 bytes for 32 elements)
} block_q4_0_t;

typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min (4 bytes)
    uint8_t qs[QK4_1 / 2];  // nibbles / quants (16 bytes)
} block_q4_1_t;

typedef struct {
    half d;                 // delta (2 bytes)
    uint8_t qh[4];          // 5-th bit of quants (4 bytes)
    uint8_t qs[QK5_0 / 2];  // nibbles / quants (16 bytes)
} block_q5_0_t;

typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min (4 bytes)
    uint8_t qh[4];          // 5-th bit of quants (4 bytes)
    uint8_t qs[QK5_1 / 2];  // nibbles / quants (16 bytes)
} block_q5_1_t;

typedef struct {
    half    d;              // delta (2 bytes)
    int8_t  qs[QK8_0];      // quants (32 bytes)
} block_q8_0_t;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K_t;

typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
    half d;                    // super-block scale
} block_q3_K_t;

typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4-bit quants
} block_q4_K_t;

typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K_t;

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;               // delta
} block_q6_K_t;

// =============================================================================
// SIMPLE FORMAT REPACKING (Q4_0, Q5_0, Q8_0, Q4_1, Q5_1)
// =============================================================================
// K/64 MODE: pack two GGML 32-element blocks into one K/64 block_c_*_k64,
//            permute quants to match loader extraction order, and embed scales.

// Unaligned load helpers (GGML blocks are not always aligned)
__device__ __forceinline__ uint16_t load_u16_unaligned(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

__device__ __forceinline__ uint32_t load_u32_unaligned(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ half load_half_unaligned(const uint8_t* p) {
    const uint16_t u = load_u16_unaligned(p);
    return *reinterpret_cast<const half*>(&u);
}

__device__ __forceinline__ half2 load_half2_unaligned(const uint8_t* p) {
    const half h0 = load_half_unaligned(p);
    const half h1 = load_half_unaligned(p + 2);
    return __halves2half2(h0, h1);
}

/// Repack Q4_0 weights to K/128 GEMX format (src -> compacted dst)
/// 
/// Q4_0 input:  [d (2B)][qs[16] (16B)] = 18 bytes per 32 elements
/// Q4_0 output: block_c_q4_0_k128 (128 elements) with embedded half scales
///              4 GGML blocks → 1 K/128 block (16 threads × 8 elements)
///              qs0..qs15 each hold 8 elements (4-bit × 8 = 32 bits = 1 int)
///              d0..d3 = scales for 4 groups of 32 elements
template <int BLOCK_SIZE>
__device__ void repack_q4_0_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;  // K/128 blocks per row
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;  // 4 GGML blocks per K/128

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q4_0;   // 18 bytes input
    constexpr int HEADER_SIZE = 2;                    // d (fp16)
    constexpr int QS_BYTES = 16;                      // 32 × 4-bit

    // Read 4 source blocks
    uint8_t src_bytes[4][QS_BYTES];
    half scales[4];
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        scales[blk] = load_half_unaligned(src_base);
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = src_base[HEADER_SIZE + i];
        }
    }

    block_c_q4_0_k128* dst = reinterpret_cast<block_c_q4_0_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q4_0_k128 out_blk;

    // Map 128 elements to 16 threads × 8 elements each
    // Thread t owns elements [t*8, t*8+7]
    // Source blocks: elements 0-31 (blk0), 32-63 (blk1), 64-95 (blk2), 96-127 (blk3)
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;  // First element for this thread
        const int src_blk = elem_base / 32;  // Which source block (0-3)
        const int in_blk = elem_base % 32;   // Position within source block
        
        // Extract 8 nibbles
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = extract_q4_0_element(src_bytes[src_blk], in_blk + i);
        }
        // Pack using LOP3-ready layout for shift-based extraction
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        
        // Store to named field (can't index, must switch)
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    // Store 4 scales (one per 32 elements = one per GGML block)
    out_blk.d0 = scales[0];
    out_blk.d1 = scales[1];
    out_blk.d2 = scales[2];
    out_blk.d3 = scales[3];

    dst[dst_idx].copy_from(out_blk);
}

/// Repack Q4_1 weights to K/128 GEMX format
/// 
/// Q4_1 input:  [dm (4B)][qs[16] (16B)] = 20 bytes per 32 elements
/// Q4_1 output: block_c_q4_1_k128 (128 elements) with embedded dm0/dm1/dm2/dm3
template <int BLOCK_SIZE>
__device__ void repack_q4_1_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q4_1;
    constexpr int HEADER_SIZE = 4;
    constexpr int QS_BYTES = 16;

    uint8_t src_bytes[4][QS_BYTES];
    half2 dm_vals[4];
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        dm_vals[blk] = load_half2_unaligned(src_base);
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = src_base[HEADER_SIZE + i];
        }
    }

    block_c_q4_1_k128* dst = reinterpret_cast<block_c_q4_1_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q4_1_k128 out_blk;

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int src_blk = elem_base / 32;
        const int in_blk = elem_base % 32;
        
        // Extract 8 nibbles for this thread
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = extract_q4_0_element(src_bytes[src_blk], in_blk + i);
        }
        
        // Pack with LOP3-ready layout for shift-based extraction
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    out_blk.dm0 = dm_vals[0];
    out_blk.dm1 = dm_vals[1];
    out_blk.dm2 = dm_vals[2];
    out_blk.dm3 = dm_vals[3];

    dst[dst_idx].copy_from(out_blk);
}

/// Repack Q5_0 weights to K/128 GEMX format
/// 
/// Q5_0 input:  [d (2B)][qh[4] (4B)][qs[16] (16B)] = 22 bytes per 32 elements
/// Q5_0 output: block_c_q5_0_k128 (128 elements) with embedded half scales
///              4 GGML blocks → 1 K/128 block
///              qs0..qs15 = low 4 bits, qh0123/qh4567/qh891011/qh12131415 = high bits
template <int BLOCK_SIZE>
__device__ void repack_q5_0_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q5_0;
    constexpr int HEADER_SIZE = 2;
    constexpr int QH_SIZE = 4;
    constexpr int QS_BYTES = 16;

    uint8_t src_bytes[4][QS_BYTES];
    uint32_t qh_vals[4];
    half scales[4];
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        scales[blk] = load_half_unaligned(src_base);
        qh_vals[blk] = load_u32_unaligned(src_base + HEADER_SIZE);
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = src_base[HEADER_SIZE + QH_SIZE + i];
        }
    }

    block_c_q5_0_k128* dst = reinterpret_cast<block_c_q5_0_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q5_0_k128 out_blk;

    // Process 4 groups of 4 threads each (each group corresponds to one GGML block)
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int src_blk = elem_base / 32;
        const int in_blk = elem_base % 32;
        
        // Extract 8 low nibbles for this thread
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = extract_q4_0_element(src_bytes[src_blk], in_blk + i);
        }
        
        // Pack low 4 bits with LOP3-ready layout for shift-based extraction
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    // Pack high bits for each group of 4 threads (each GGML block has 32 high bits)
    // Thread t in group g needs bits for elements [t*8, t*8+7] within that block
    // Q5_0 qh layout: bit i of qh = high bit for element i (i=0..31)
    auto pack_qh_for_group = [&](int grp) -> int {
        uint8_t qh_packed[4];
        for (int local_t = 0; local_t < 4; local_t++) {
            const int in_blk = local_t * 8;  // Element offset within block
            uint8_t bits = 0;
            for (int b = 0; b < 8; b++) {
                const int elem = in_blk + b;
                const uint8_t hbit = (qh_vals[grp] >> elem) & 1;
                bits |= (hbit << b);
            }
            qh_packed[local_t] = bits;
        }
        return *reinterpret_cast<int*>(qh_packed);
    };

    out_blk.qh0123 = pack_qh_for_group(0);
    out_blk.qh4567 = pack_qh_for_group(1);
    out_blk.qh891011 = pack_qh_for_group(2);
    out_blk.qh12131415 = pack_qh_for_group(3);

    out_blk.d0 = scales[0];
    out_blk.d1 = scales[1];
    out_blk.d2 = scales[2];
    out_blk.d3 = scales[3];

    dst[dst_idx].copy_from(out_blk);
}

/// Repack Q5_1 weights to K/128 GEMX format
/// 
/// Q5_1 input:  [dm (4B)][qh[4] (4B)][qs[16] (16B)] = 24 bytes per 32 elements
/// Q5_1 output: block_c_q5_1_k128 (128 elements) with embedded dm0/dm1/dm2/dm3
template <int BLOCK_SIZE>
__device__ void repack_q5_1_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q5_1;
    constexpr int HEADER_SIZE = 4;
    constexpr int QH_SIZE = 4;
    constexpr int QS_BYTES = 16;

    uint8_t src_bytes[4][QS_BYTES];
    uint32_t qh_vals[4];
    half2 dm_vals[4];
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        dm_vals[blk] = load_half2_unaligned(src_base);
        qh_vals[blk] = load_u32_unaligned(src_base + HEADER_SIZE);
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = src_base[HEADER_SIZE + QH_SIZE + i];
        }
    }

    block_c_q5_1_k128* dst = reinterpret_cast<block_c_q5_1_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q5_1_k128 out_blk;

    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int src_blk = elem_base / 32;
        const int in_blk = elem_base % 32;
        
        // Extract 8 consecutive 4-bit low nibbles
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = extract_q4_0_element(src_bytes[src_blk], in_blk + i);
        }
        
        // Pack using LOP3-ready layout:
        // bits[3:0]=n0, [7:4]=n4, [11:8]=n2, [15:12]=n6
        // bits[19:16]=n1, [23:20]=n5, [27:24]=n3, [31:28]=n7
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    // Pack high bits for each group of 4 threads
    auto pack_qh_for_group = [&](int grp) -> int {
        uint8_t qh_packed[4];
        for (int local_t = 0; local_t < 4; local_t++) {
            const int in_blk = local_t * 8;
            uint8_t bits = 0;
            for (int b = 0; b < 8; b++) {
                const int elem = in_blk + b;
                const uint8_t hbit = (qh_vals[grp] >> elem) & 1;
                bits |= (hbit << b);
            }
            qh_packed[local_t] = bits;
        }
        return *reinterpret_cast<int*>(qh_packed);
    };

    out_blk.qh0123 = pack_qh_for_group(0);
    out_blk.qh4567 = pack_qh_for_group(1);
    out_blk.qh891011 = pack_qh_for_group(2);
    out_blk.qh12131415 = pack_qh_for_group(3);

    out_blk.dm0 = dm_vals[0];
    out_blk.dm1 = dm_vals[1];
    out_blk.dm2 = dm_vals[2];
    out_blk.dm3 = dm_vals[3];

    dst[dst_idx].copy_from(out_blk);
}

/// Repack Q8_0 weights to K/128 GEMX format
/// 
/// Q8_0 input:  [d (2B)][qs[32] (32B)] = 34 bytes per 32 elements
/// Q8_0 output: block_c_q8_0_k128 (128 elements) with embedded half scales
///              4 GGML blocks → 1 K/128 block (16 threads × 8 elements)
///              qs0..qs15 each hold int2 (8 × int8)
template <int BLOCK_SIZE>
__device__ void repack_q8_0_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q8_0;
    constexpr int HEADER_SIZE = 2;
    constexpr int QS_BYTES = 32;

    int8_t src_bytes[4][QS_BYTES];
    half scales[4];
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        scales[blk] = load_half_unaligned(src_base);
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = static_cast<int8_t>(src_base[HEADER_SIZE + i]);
        }
    }

    block_c_q8_0_k128* dst = reinterpret_cast<block_c_q8_0_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q8_0_k128 out_blk;

    // Each thread owns 8 consecutive int8 values
    // Thread t owns elements [t*8, t*8+7] from the 128 total
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int src_blk = elem_base / 32;
        const int in_blk = elem_base % 32;
        
        // Get 8 consecutive bytes as int2
        int2 qs_val;
        qs_val.x = *reinterpret_cast<const int*>(&src_bytes[src_blk][in_blk]);
        qs_val.y = *reinterpret_cast<const int*>(&src_bytes[src_blk][in_blk + 4]);
        
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    out_blk.d0 = scales[0];
    out_blk.d1 = scales[1];
    out_blk.d2 = scales[2];
    out_blk.d3 = scales[3];

    dst[dst_idx].copy_from(out_blk);
}

/// Repack Q8_1 weights to K/128 GEMX format
/// 
/// Q8_1 input:  [ds (4B half2)][qs[32] (32B)] = 36 bytes per 32 elements
///              ds.x = delta (scale), ds.y = sum
/// Q8_1 output: block_c_q8_1_k128 (128 elements) with embedded half2 dm pairs
///              4 GGML blocks → 1 K/128 block (16 threads × 8 elements)
///              qs0..qs15 each hold int2 (8 × int8)
template <int BLOCK_SIZE>
__device__ void repack_q8_1_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;
    const int col128 = block_idx % blocks_per_row;

    const int src_blocks_per_row = ncols / 32;
    const int src_block_base = row * src_blocks_per_row + col128 * 4;

    constexpr int SRC_BLOCK_SIZE = BLOCK_SIZE_Q8_1;  // 36 bytes
    constexpr int HEADER_SIZE = 4;  // half2 ds
    constexpr int QS_BYTES = 32;

    int8_t src_bytes[4][QS_BYTES];
    half2 dm_vals[4];  // dm.x = d (scale), dm.y = m (sum)
    
    #pragma unroll
    for (int blk = 0; blk < 4; blk++) {
        const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                                  (src_block_base + blk) * SRC_BLOCK_SIZE;
        dm_vals[blk] = *reinterpret_cast<const half2*>(src_base);  // Load half2 ds
        #pragma unroll
        for (int i = 0; i < QS_BYTES; i++) {
            src_bytes[blk][i] = static_cast<int8_t>(src_base[HEADER_SIZE + i]);
        }
    }

    block_c_q8_1_k128* dst = reinterpret_cast<block_c_q8_1_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q8_1_k128 out_blk;

    // Each thread owns 8 consecutive int8 values
    // Thread t owns elements [t*8, t*8+7] from the 128 total
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int src_blk = elem_base / 32;
        const int in_blk = elem_base % 32;
        
        // Get 8 consecutive bytes as int2
        int2 qs_val;
        qs_val.x = *reinterpret_cast<const int*>(&src_bytes[src_blk][in_blk]);
        qs_val.y = *reinterpret_cast<const int*>(&src_bytes[src_blk][in_blk + 4]);
        
        switch(t) {
            case 0: out_blk.qs0 = qs_val; break;
            case 1: out_blk.qs1 = qs_val; break;
            case 2: out_blk.qs2 = qs_val; break;
            case 3: out_blk.qs3 = qs_val; break;
            case 4: out_blk.qs4 = qs_val; break;
            case 5: out_blk.qs5 = qs_val; break;
            case 6: out_blk.qs6 = qs_val; break;
            case 7: out_blk.qs7 = qs_val; break;
            case 8: out_blk.qs8 = qs_val; break;
            case 9: out_blk.qs9 = qs_val; break;
            case 10: out_blk.qs10 = qs_val; break;
            case 11: out_blk.qs11 = qs_val; break;
            case 12: out_blk.qs12 = qs_val; break;
            case 13: out_blk.qs13 = qs_val; break;
            case 14: out_blk.qs14 = qs_val; break;
            case 15: out_blk.qs15 = qs_val; break;
        }
    }

    // Store dm pairs (half2 with d=scale, m=sum)
    out_blk.dm0 = dm_vals[0];
    out_blk.dm1 = dm_vals[1];
    out_blk.dm2 = dm_vals[2];
    out_blk.dm3 = dm_vals[3];

    dst[dst_idx].copy_from(out_blk);
}

// =============================================================================
// Q8_K REPACKING (K-QUANT 8-BIT)
// =============================================================================
// Q8_K input:  float d, int8_t qs[256], int16_t bsums[16] = 292 bytes per 256 elems
// Q8_K output: block_c_q8_K_k128 = 160 bytes per 128 elems
//              Same layout as Q8_1 K/128 but scale comes from float d (shared)
//
// Each GGML Q8_K block (256 elements) becomes 2 K/128 blocks (128 elems each).
// One CUDA block processes one K/128 output block (warp-level).
__device__ void repack_q8_K_impl(
    const void* __restrict__ src_data,
    void* __restrict__ dst_data,
    int nrows,
    int ncols
) {
    // Each CUDA block handles one K/128 output block
    const int k128_blocks_per_row = ncols / 128;
    const int total_k128_blocks = nrows * k128_blocks_per_row;
    
    const int block_idx = blockIdx.x;
    if (block_idx >= total_k128_blocks) return;
    
    const int row = block_idx / k128_blocks_per_row;
    const int col128 = block_idx % k128_blocks_per_row;
    const int half_idx = col128 % 2;  // 0 = first 128 elems, 1 = second 128 elems
    
    // Find source Q8_K super-block (256 elements each)
    const int src_blocks_per_row = ncols / 256;
    const int src_block_idx = row * src_blocks_per_row + col128 / 2;
    
    // Q8_K layout: float d (4B) + int8_t qs[256] + int16_t bsums[16]
    constexpr int Q8K_D_OFFSET = 0;
    constexpr int Q8K_QS_OFFSET = 4;
    
    const uint8_t* src_base = reinterpret_cast<const uint8_t*>(src_data) + 
                              src_block_idx * BLOCK_SIZE_Q8_K;
    
    // Load scale d (float32 -> half)
    float d_f32 = *reinterpret_cast<const float*>(src_base + Q8K_D_OFFSET);
    half d_h = __float2half(d_f32);
    half2 d_half2 = make_half2(d_h, __float2half(0.0f));  // d0.x = d, d0.y = unused
    
    // Load 128 quant bytes for this half (threadIdx.x determines which bytes)
    const int qs_base = Q8K_QS_OFFSET + half_idx * 128;
    
    // Each of 32 threads loads bytes in a strided pattern to cover 128 bytes
    // But output needs 16 threads × 8 bytes each
    // Use first 16 threads to build the output
    
    int8_t qs_local[8];
    if (threadIdx.x < 16) {
        const int t = threadIdx.x;
        const int elem_base = t * 8;  // Each thread owns 8 consecutive elements
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            qs_local[i] = static_cast<int8_t>(src_base[qs_base + elem_base + i]);
        }
    }
    
    // Build output block using thread 0
    block_c_q8_K_k128* dst = reinterpret_cast<block_c_q8_K_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;  // Column-major layout
    
    // Use shared memory to gather all threads' data
    __shared__ int2 shared_qs[16];
    
    if (threadIdx.x < 16) {
        int2 qs_val;
        qs_val.x = *reinterpret_cast<const int*>(&qs_local[0]);
        qs_val.y = *reinterpret_cast<const int*>(&qs_local[4]);
        shared_qs[threadIdx.x] = qs_val;
    }
    __syncthreads();
    
    // Thread 0 writes the output block
    if (threadIdx.x == 0) {
        block_c_q8_K_k128 out_blk;
        
        out_blk.qs0 = shared_qs[0];
        out_blk.qs1 = shared_qs[1];
        out_blk.qs2 = shared_qs[2];
        out_blk.qs3 = shared_qs[3];
        out_blk.qs4 = shared_qs[4];
        out_blk.qs5 = shared_qs[5];
        out_blk.qs6 = shared_qs[6];
        out_blk.qs7 = shared_qs[7];
        out_blk.qs8 = shared_qs[8];
        out_blk.qs9 = shared_qs[9];
        out_blk.qs10 = shared_qs[10];
        out_blk.qs11 = shared_qs[11];
        out_blk.qs12 = shared_qs[12];
        out_blk.qs13 = shared_qs[13];
        out_blk.qs14 = shared_qs[14];
        out_blk.qs15 = shared_qs[15];
        
        // All 4 scale slots get the same d value (shared across entire K/128 block)
        out_blk.d0 = d_half2;
        out_blk.d1 = d_half2;
        out_blk.d2 = d_half2;
        out_blk.d3 = d_half2;
        out_blk._pad0 = make_half2(__float2half(0.0f), __float2half(0.0f));
        out_blk._pad1 = make_half2(__float2half(0.0f), __float2half(0.0f));
        
        dst[dst_idx].copy_from(out_blk);
    }
}
