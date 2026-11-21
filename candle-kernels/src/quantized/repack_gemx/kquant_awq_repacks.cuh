// The K-quant and AWQ repacks, with the size helpers.
//
// The `q2k_extract_crumb` helper and the repacks `repack_q2_K_impl`,
// `repack_q3_K_impl`, `repack_q4_K_impl`, `repack_q5_K_impl` and
// `repack_q6_K_impl` with their element extractors; `get_repacked_size` and
// `get_input_size`; and the AWQ repacks `repack_q_awq_impl`,
// `repack_q_awq_g64_impl` and `repack_q_awq_hf_impl`.

// =============================================================================
// K-QUANT FORMAT REPACKING (Q2_K - Q6_K)
// =============================================================================
// K-quants have 256-element super-blocks with embedded sub-block scales.
// These are more complex - scales are interleaved with quants.
// Use warp-level processing with shared memory for coalesced access.

// Q4_K GEMX repacking is defined in loader/q4_K.cuh (gemx namespace)

// Q2K/Q3K K-tile permutation: Maps dst crumb position -> src element within a 16-element tile
// Based on GEMX extraction shifts for 2-bit:
//   GEMX extracts crumbs at bits {0,4,2,6, 8,12,10,14, 16,20,18,22, 24,28,26,30}
//   This gives element order: 0,2,1,3, 4,6,5,7, 8,10,9,11, 12,14,13,15
__constant__ uint8_t Q2K_TILE_PERM[16] = {
    0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15
};

/// Extract 2-bit crumb from Q2_K/Q3_K qs array given element index (0-255)
/// GGML K-quant 2-bit layout (same for Q2_K and Q3_K):
///   256 elements split into 2 chunks of 128, each using 32 bytes of qs
///   Within each 128-element chunk:
///     - Elements 0-15:   qs[0..15]  >> 0 (bits 0-1)
///     - Elements 16-31:  qs[16..31] >> 0 (bits 0-1)
///     - Elements 32-47:  qs[0..15]  >> 2 (bits 2-3)
///     - Elements 48-63:  qs[16..31] >> 2 (bits 2-3)
///     - Elements 64-79:  qs[0..15]  >> 4 (bits 4-5)
///     - Elements 80-95:  qs[16..31] >> 4 (bits 4-5)
///     - Elements 96-111: qs[0..15]  >> 6 (bits 6-7)
///     - Elements 112-127: qs[16..31] >> 6 (bits 6-7)
///   Second chunk (elements 128-255) uses qs[32..63] with same pattern.
///
/// Formula: element e → qs[(e/128)*32 + (e%32)] >> (((e%128)/32)*2)
__device__ __forceinline__ uint8_t q2k_extract_crumb(const uint8_t* qs, int elem) {
    const int chunk128 = elem >> 7;                    // elem / 128 (0 or 1)
    const int byte_in_chunk = elem & 31;               // elem % 32
    const int byte_idx = chunk128 * 32 + byte_in_chunk;
    const int shift = ((elem & 127) >> 5) << 1;        // ((elem % 128) / 32) * 2
    return (qs[byte_idx] >> shift) & 0x3;
}

/// Repack Q2_K weights to K/128 format (src -> dst)
/// 
/// Q2_K input:  [N, K/QK_K, 84B] = row-major blocks (QK_K=256)
/// Q2_K output: [K/128, N] block_c_q2_K_k128 with embedded half2 scales
///              Each 256-elem superblock produces 2 K/128 blocks
///              16 threads × 8 elements = 128 elements
///              8 scales (one per 16 elements)
__device__ void repack_q2_K_impl(
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

    const int superblocks_per_row = ncols / QK_K;
    const int k128_per_superblock = QK_K / 128;  // 2
    const int superblock_col = col128 / k128_per_superblock;
    const int block_offset = col128 % k128_per_superblock;  // 0 or 1
    const int superblock_idx = row * superblocks_per_row + superblock_col;

    const block_q2_K_t* src_blocks = reinterpret_cast<const block_q2_K_t*>(src_data);
    const block_q2_K_t* src = &src_blocks[superblock_idx];

    block_c_q2_K_k128 out_blk;

    const float2 dm = __half22float2(src->dm);
    const float d = dm.x;
    const float dmin = dm.y;

    // 16 threads, 8 elements each
    // Thread t owns elements [base + t*8, base + t*8+7] where base = block_offset * 128
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = block_offset * 128 + t * 8;
        
        // Pack 8 crumbs (2-bit) into 16 bits
        uint16_t qs_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t crumb = q2k_extract_crumb(src->qs, elem_base + i);
            qs_packed |= (static_cast<uint16_t>(crumb) << (i * 2));
        }
        
        switch(t) {
            case 0: out_blk.qs0 = qs_packed; break;
            case 1: out_blk.qs1 = qs_packed; break;
            case 2: out_blk.qs2 = qs_packed; break;
            case 3: out_blk.qs3 = qs_packed; break;
            case 4: out_blk.qs4 = qs_packed; break;
            case 5: out_blk.qs5 = qs_packed; break;
            case 6: out_blk.qs6 = qs_packed; break;
            case 7: out_blk.qs7 = qs_packed; break;
            case 8: out_blk.qs8 = qs_packed; break;
            case 9: out_blk.qs9 = qs_packed; break;
            case 10: out_blk.qs10 = qs_packed; break;
            case 11: out_blk.qs11 = qs_packed; break;
            case 12: out_blk.qs12 = qs_packed; break;
            case 13: out_blk.qs13 = qs_packed; break;
            case 14: out_blk.qs14 = qs_packed; break;
            case 15: out_blk.qs15 = qs_packed; break;
        }
    }

    // 8 scales for 128 elements (one per 16 elements)
    // Scales are packed: each byte has 4-bit scale and 4-bit min
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int scale_idx = block_offset * 8 + s;
        const uint8_t sc_byte = src->scales[scale_idx];
        const int scale_4bit = sc_byte & 0xF;
        const int min_4bit = sc_byte >> 4;
        const float scale = d * float(scale_4bit);
        const float neg_min = -dmin * float(min_4bit);
        
        switch(s) {
            case 0: out_blk.dm0 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 1: out_blk.dm1 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 2: out_blk.dm2 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 3: out_blk.dm3 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 4: out_blk.dm4 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 5: out_blk.dm5 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 6: out_blk.dm6 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 7: out_blk.dm7 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
        }
    }

    const int dst_idx = col128 * nrows + row;
    reinterpret_cast<block_c_q2_K_k128*>(dst_data)[dst_idx].copy_from(out_blk);
}

/// Repack Q3_K weights to K/128 format (src -> dst)
///
/// Q3_K input:  [N, K/QK_K, 110B] = row-major blocks (QK_K=256)
/// Q3_K output: [K/128, N] block_c_q3_K_k128 with embedded scales
///              Each 256-elem superblock produces 2 K/128 blocks
///              16 threads × 8 elements = 128 elements
///              8 scales (one per 16 elements)
__device__ void repack_q3_K_impl(
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

    const int superblocks_per_row = ncols / QK_K;
    const int k128_per_superblock = QK_K / 128;  // 2
    const int superblock_col = col128 / k128_per_superblock;
    const int block_offset = col128 % k128_per_superblock;  // 0 or 1
    const int superblock_idx = row * superblocks_per_row + superblock_col;

    const block_q3_K_t* src_blocks = reinterpret_cast<const block_q3_K_t*>(src_data);
    const block_q3_K_t* src = &src_blocks[superblock_idx];

    block_c_q3_K_k128 out_blk;

    const float d = __half2float(src->d);

    // Decode scales for this 128-element block first
    constexpr uint32_t KMASK1 = 0x03030303;
    constexpr uint32_t KMASK2 = 0x0f0f0f0f;
    const uint8_t* raw = src->scales;
    const uint32_t aux0 = raw[0] | (raw[1] << 8) | (raw[2] << 16) | (raw[3] << 24);
    const uint32_t aux1 = raw[4] | (raw[5] << 8) | (raw[6] << 16) | (raw[7] << 24);
    const uint32_t tmp  = raw[8] | (raw[9] << 8) | (raw[10] << 16) | (raw[11] << 24);
    uint32_t new_aux[4];
    new_aux[0] = (aux0 & KMASK2) | (((tmp) & KMASK1) << 4);
    new_aux[1] = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    new_aux[2] = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    new_aux[3] = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);

    // Helper lambda to get 6-bit scale for a 16-element group
    auto get_scale = [&](int scale_idx) -> float {
        const int aux_idx = scale_idx / 4;
        const int byte_idx = scale_idx % 4;
        const int8_t scale_signed = static_cast<int8_t>((new_aux[aux_idx] >> (byte_idx * 8)) & 0xFF);
        return d * (float(scale_signed) - 32.0f);
    };

    // Helper lambda to get hmask bit for element
    auto get_hmask_bit = [&](int elem) -> uint8_t {
        const int byte_idx = elem & 31;
        const int bit_pos = elem >> 5;
        return (src->hmask[byte_idx] >> bit_pos) & 1;
    };

    // 16 threads, 8 elements each
    // Thread t owns elements [base + t*8, base + t*8+7] where base = block_offset * 128
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = block_offset * 128 + t * 8;
        
        // Pack 8 crumbs (2-bit low) into 16 bits
        uint16_t qs_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t crumb = q2k_extract_crumb(src->qs, elem_base + i);
            qs_packed |= (static_cast<uint16_t>(crumb) << (i * 2));
        }
        
        // Pack 8 high bits into uint8_t
        uint8_t qh_packed = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t hbit = get_hmask_bit(elem_base + i);
            qh_packed |= (hbit << i);
        }
        
        // Store to named fields - must use switch statements
        switch(t) {
            case 0: out_blk.qs0 = qs_packed; out_blk.qh0 = qh_packed; break;
            case 1: out_blk.qs1 = qs_packed; out_blk.qh1 = qh_packed; break;
            case 2: out_blk.qs2 = qs_packed; out_blk.qh2 = qh_packed; break;
            case 3: out_blk.qs3 = qs_packed; out_blk.qh3 = qh_packed; break;
            case 4: out_blk.qs4 = qs_packed; out_blk.qh4 = qh_packed; break;
            case 5: out_blk.qs5 = qs_packed; out_blk.qh5 = qh_packed; break;
            case 6: out_blk.qs6 = qs_packed; out_blk.qh6 = qh_packed; break;
            case 7: out_blk.qs7 = qs_packed; out_blk.qh7 = qh_packed; break;
            case 8: out_blk.qs8 = qs_packed; out_blk.qh8 = qh_packed; break;
            case 9: out_blk.qs9 = qs_packed; out_blk.qh9 = qh_packed; break;
            case 10: out_blk.qs10 = qs_packed; out_blk.qh10 = qh_packed; break;
            case 11: out_blk.qs11 = qs_packed; out_blk.qh11 = qh_packed; break;
            case 12: out_blk.qs12 = qs_packed; out_blk.qh12 = qh_packed; break;
            case 13: out_blk.qs13 = qs_packed; out_blk.qh13 = qh_packed; break;
            case 14: out_blk.qs14 = qs_packed; out_blk.qh14 = qh_packed; break;
            case 15: out_blk.qs15 = qs_packed; out_blk.qh15 = qh_packed; break;
        }
    }

    // 8 scales for 128 elements (one per 16 elements)
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int scale_idx = block_offset * 8 + s;
        const float scale = get_scale(scale_idx);
        const half2 dm = __halves2half2(__float2half(scale), __float2half(0.0f));
        
        switch(s) {
            case 0: out_blk.dm0 = dm; break;
            case 1: out_blk.dm1 = dm; break;
            case 2: out_blk.dm2 = dm; break;
            case 3: out_blk.dm3 = dm; break;
            case 4: out_blk.dm4 = dm; break;
            case 5: out_blk.dm5 = dm; break;
            case 6: out_blk.dm6 = dm; break;
            case 7: out_blk.dm7 = dm; break;
        }
    }

    const int dst_idx = col128 * nrows + row;
    reinterpret_cast<block_c_q3_K_k128*>(dst_data)[dst_idx].copy_from(out_blk);
}

/// Repack Q4_K weights to K/128 format (src -> dst)
/// 
/// Q4_K input:  [N, K/QK_K, 144B] = row-major blocks (QK_K=256)
/// Q4_K output: [K/128, N] block_c_q4_K_k128 with embedded scale/min pairs
///              Each 256-elem superblock produces 2 K/128 blocks
///              16 threads × 8 elements = 128 elements
///              4 scales (one per 32 elements)

/// Extract nibble from Q4_K qs array given element index (0-255)
/// Q4_K nibble layout:
///   Elements 0-63:   bytes 0-63 low nibble
///   Elements 64-127: bytes 0-63 high nibble
///   Elements 128-191: bytes 64-127 low nibble
///   Elements 192-255: bytes 64-127 high nibble
// Q4_K element extraction from qs array
// Layout: 4 groups of 32 bytes, each covering 64 elements
//   - Within each 32-byte group:
//     * Low nibbles (q & 0xF) → elements 0-31 (scale j*2)
//     * High nibbles (q >> 4) → elements 32-63 (scale j*2+1)
__device__ __forceinline__ uint8_t q4k_extract_element(const uint8_t* qs, int elem) {
    const int j = elem / 64;           // Which 64-element group (0-3)
    const int in_group = elem % 64;    // Position within 64-element group (0-63)
    const int is_high = in_group / 32; // 0 = low nibble, 1 = high nibble
    const int byte_offset = in_group % 32;  // Byte within 32-byte section
    const int byte_idx = j * 32 + byte_offset;
    const uint8_t byte_val = qs[byte_idx];
    return is_high ? (byte_val >> 4) : (byte_val & 0x0F);
}

__device__ void repack_q4_K_impl(
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

    const int superblocks_per_row = ncols / QK_K;
    const int k128_per_superblock = QK_K / 128;  // 2
    const int superblock_col = col128 / k128_per_superblock;
    const int block_offset = col128 % k128_per_superblock;  // 0 or 1
    const int superblock_idx = row * superblocks_per_row + superblock_col;

    const block_q4_K_t* src_blocks = reinterpret_cast<const block_q4_K_t*>(src_data);
    const block_q4_K_t* src = &src_blocks[superblock_idx];

    block_c_q4_K_k128 out_blk;

    // 16 threads, 8 elements each
    // Thread t owns elements [base + t*8, base + t*8+7] where base = block_offset * 128
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = block_offset * 128 + t * 8;
        
        // Extract 8 nibbles and pack with LOP3-ready layout for fast shift-based extraction
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = q4k_extract_element(src->qs, elem_base + i);
        }
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

    // Decode 4 scale/min pairs for this 128-element block (one per 32 elements)
    const float2 dm = __half22float2(src->dm);
    const float d = dm.x;
    const float dmin = dm.y;
    const uint8_t* sc = src->scales;

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int sub_idx = block_offset * 4 + s;  // 0-3 for block_offset=0, 4-7 for block_offset=1
        int scale_6bit, min_6bit;
        if (sub_idx < 4) {
            scale_6bit = sc[sub_idx] & 0x3F;
            min_6bit = sc[sub_idx + 4] & 0x3F;
        } else {
            scale_6bit = ((sc[sub_idx + 4] & 0x0F) | ((sc[sub_idx - 4] >> 6) << 4));
            min_6bit = ((sc[sub_idx + 4] >> 4) | ((sc[sub_idx] >> 6) << 4));
        }
        const float scale = d * float(scale_6bit);
        const float neg_min = -dmin * float(min_6bit);
        
        switch(s) {
            case 0: out_blk.dm0 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 1: out_blk.dm1 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 2: out_blk.dm2 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 3: out_blk.dm3 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
        }
    }

    const int dst_idx = col128 * nrows + row;
    reinterpret_cast<block_c_q4_K_k128*>(dst_data)[dst_idx].copy_from(out_blk);
}

/// Repack Q5_K weights to K/128 format (src -> dst)
/// 
/// Q5_K input:  [N, K/QK_K, 176B] = row-major blocks (QK_K=256)
/// Q5_K output: [K/128, N] block_c_q5_K_k128 with embedded scale/min pairs
///              Each 256-elem superblock produces 2 K/128 blocks
///              16 threads × 8 elements = 128 elements
///              4 scales (one per 32 elements)

/// Extract 4-bit low nibble from Q5_K qs array given element index (0-255)
/// Q5_K qs layout is BLOCK INTERLEAVED (different from Q4_K!):
///   Elements 0-31:    qs[0-31] low nibbles
///   Elements 32-63:   qs[0-31] high nibbles
///   Elements 64-95:   qs[32-63] low nibbles
///   Elements 96-127:  qs[32-63] high nibbles
///   Elements 128-159: qs[64-95] low nibbles
///   Elements 160-191: qs[64-95] high nibbles
///   Elements 192-223: qs[96-127] low nibbles
///   Elements 224-255: qs[96-127] high nibbles
__device__ __forceinline__ uint8_t q5k_extract_ql(const uint8_t* qs, int elem) {
    // Q5_K uses 32-element interleaving: 32 low, 32 high, repeat
    const int block_64 = elem >> 6;       // 0-3: which 64-element block
    const int in_block = elem & 63;       // 0-63: position within block
    const int is_high = (in_block >= 32); // High nibble for elements 32-63 in each block
    const int byte_idx = block_64 * 32 + (in_block & 31);  // Map to bytes 0-127
    const uint8_t byte_val = qs[byte_idx];
    return is_high ? (byte_val >> 4) : (byte_val & 0x0F);
}

/// Extract 1-bit high bit from Q5_K qh array given element index (0-255)
/// Q5_K qh layout: 32 bytes, bit j of qh[i] = high bit for element (32*j + i)
/// So element e uses qh[e % 32] bit (e / 32)
__device__ __forceinline__ uint8_t q5k_extract_qh(const uint8_t* qh, int elem) {
    // GGML Q5_K layout: qh[i] holds bits for elements i, 32+i, 64+i, etc.
    const int byte_idx = elem & 31;        // elem % 32
    const int bit_idx = elem >> 5;         // elem / 32 (0-7 for elements 0-255)
    return (qh[byte_idx] >> bit_idx) & 1;
}

__device__ void repack_q5_K_impl(
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

    const int superblocks_per_row = ncols / QK_K;
    const int k128_per_superblock = QK_K / 128;  // 2
    const int superblock_col = col128 / k128_per_superblock;
    const int block_offset = col128 % k128_per_superblock;  // 0 or 1
    const int superblock_idx = row * superblocks_per_row + superblock_col;

    const block_q5_K_t* src_blocks = reinterpret_cast<const block_q5_K_t*>(src_data);
    const block_q5_K_t* src = &src_blocks[superblock_idx];

    block_c_q5_K_k128 out_blk;

    // 16 threads, 8 elements each
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = block_offset * 128 + t * 8;
        
        // Extract 8 nibbles (4-bit each)
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = q5k_extract_ql(src->qs, elem_base + i);
        }
        // Pack using LOP3-ready layout for shift-based extraction
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

    // Pack high bits for each group of 4 threads (each group = 32 elements)
    auto pack_qh_for_group = [&](int grp) -> int {
        uint8_t qh_packed[4];
        for (int local_t = 0; local_t < 4; local_t++) {
            const int elem_base = block_offset * 128 + grp * 32 + local_t * 8;
            uint8_t bits = 0;
            for (int b = 0; b < 8; b++) {
                const uint8_t hbit = q5k_extract_qh(src->qh, elem_base + b);
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

    // Decode 4 scale/min pairs for this 128-element block (one per 32 elements)
    const float2 dm = __half22float2(src->dm);
    const float d = dm.x;
    const float dmin = dm.y;
    const uint8_t* sc = src->scales;

    #pragma unroll
    for (int s = 0; s < 4; s++) {
        const int sub_idx = block_offset * 4 + s;
        int scale_6bit, min_6bit;
        if (sub_idx < 4) {
            scale_6bit = sc[sub_idx] & 0x3F;
            min_6bit = sc[sub_idx + 4] & 0x3F;
        } else {
            scale_6bit = ((sc[sub_idx + 4] & 0x0F) | ((sc[sub_idx - 4] >> 6) << 4));
            min_6bit = ((sc[sub_idx + 4] >> 4) | ((sc[sub_idx] >> 6) << 4));
        }
        const float scale = d * float(scale_6bit);
        const float neg_min = -dmin * float(min_6bit);
        
        switch(s) {
            case 0: out_blk.dm0 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 1: out_blk.dm1 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 2: out_blk.dm2 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
            case 3: out_blk.dm3 = __halves2half2(__float2half(scale), __float2half(neg_min)); break;
        }
    }

    const int dst_idx = col128 * nrows + row;
    reinterpret_cast<block_c_q5_K_k128*>(dst_data)[dst_idx].copy_from(out_blk);
}

/// Repack Q6_K weights to K/128 format (src -> dst)
/// 
/// Q6_K input:  [N, K/QK_K, 210B] = row-major blocks (QK_K=256)
/// Q6_K output: [K/128, N] block_c_q6_K_k128 with embedded scales
///              Each 256-elem superblock produces 2 K/128 blocks
///              16 threads × 8 elements = 128 elements
///              8 scales (one per 16 elements)
///
/// COMPACT 112-BYTE LAYOUT:
///   Bytes 0-63:   ql[16]    - contiguous nibbles for perfect coalescing
///   Bytes 64-95:  qh[8]     - packed crumbs (qh_lo | qh_hi << 16)
///   Bytes 96-111: scales[8] - half scales

/// Extract 6-bit element from Q6_K ql/qh arrays given element index (0-255)
/// Q6_K layout:
///   ql[128]: 256 × 4-bit low nibbles
///     Elements 0-63:   ql[0..31] low nibble, ql[0..31] high nibble  
///     Elements 64-127: ql[32..63] low nibble, ql[32..63] high nibble
///     Elements 128-191: ql[64..95] low nibble, ql[64..95] high nibble
///     Elements 192-255: ql[96..127] low nibble, ql[96..127] high nibble
///   qh[64]: 256 × 2-bit high crumbs
///     4 crumbs per byte, sequential
__device__ __forceinline__ uint8_t q6k_extract_ql(const uint8_t* ql, int elem) {
    // Q6_K ql layout (128 bytes for 256 elements):
    // Elements 0-63: bytes 0-63, where elem k uses byte k  
    //   - Elements 0-31 (half 0): bytes 0-31 LOW nibbles
    //   - Elements 64-95 (half 1): bytes 0-31 HIGH nibbles  
    // Elements 128-255: bytes 64-127, where elem 128+k uses byte 64+k
    //   - Elements 128-159 (half 2): bytes 64-95 LOW nibbles
    //   - Elements 192-223 (half 3): bytes 64-95 HIGH nibbles
    //
    // Pattern: half 0,2 = LOW nibbles, half 1,3 = HIGH nibbles
    const int half = elem >> 6;           // 0-3: which 64-element half
    const int in_half = elem & 63;        // 0-63: position within half
    const int is_high_nibble = half & 1;  // Odd halves use HIGH nibble
    const int byte_idx = (half >> 1) * 64 + in_half;  // Map to bytes 0-127
    const uint8_t byte_val = ql[byte_idx];
    return is_high_nibble ? (byte_val >> 4) : (byte_val & 0x0F);
}

__device__ __forceinline__ uint8_t q6k_extract_qh(const uint8_t* qh, int elem) {
    // Q6_K qh layout (64 bytes for 256 elements):
    // Each byte contains 4 crumbs for 4 elements that are 32 apart:
    //   - Elements 0-31: bytes 0-31, where elem k uses byte k, crumb 0 (bits 1:0)
    //   - Elements 32-63: bytes 0-31, where elem 32+k uses byte k, crumb 1 (bits 3:2)
    //   - Elements 64-95: bytes 0-31, where elem 64+k uses byte k, crumb 2 (bits 5:4)
    //   - Elements 96-127: bytes 0-31, where elem 96+k uses byte k, crumb 3 (bits 7:6)
    //   - Elements 128-159: bytes 32-63, crumb 0
    //   - Elements 160-191: bytes 32-63, crumb 1
    //   - Elements 192-223: bytes 32-63, crumb 2
    //   - Elements 224-255: bytes 32-63, crumb 3
    const int half = elem >> 7;           // 0-1: which 128-element half
    const int in_half = elem & 127;       // 0-127: position within half
    const int crumb = in_half >> 5;       // 0-3: which crumb in the byte
    const int in_crumb = in_half & 31;    // 0-31: position within 32-element group
    const int byte_idx = half * 32 + in_crumb;  // bytes 0-63
    const int crumb_pos = crumb << 1;     // 0, 2, 4, or 6
    return (qh[byte_idx] >> crumb_pos) & 0x3;
}

__device__ void repack_q6_K_impl(
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

    const int superblocks_per_row = ncols / QK_K;
    const int k128_per_superblock = QK_K / 128;  // 2
    const int superblock_col = col128 / k128_per_superblock;
    const int block_offset = col128 % k128_per_superblock;  // 0 or 1
    const int superblock_idx = row * superblocks_per_row + superblock_col;

    const block_q6_K_t* src_blocks = reinterpret_cast<const block_q6_K_t*>(src_data);
    const block_q6_K_t* src = &src_blocks[superblock_idx];

    block_c_q6_K_k128 out_blk;

    const float d = __half2float(src->d);

    // =========================================================================
    // COMPACT 112-BYTE LAYOUT: Contiguous arrays for better coalescing
    // =========================================================================
    
    // PASS 1: Pack ql[16] - one int per thread (bytes 0-63)
    // 16 threads, 8 elements each
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = block_offset * 128 + t * 8;
        
        // Extract 8 nibbles (4-bit each)
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            n[i] = q6k_extract_ql(src->ql, elem_base + i);
        }
        // Pack using LOP3-ready layout for shift-based extraction
        out_blk.ql[t] = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
    }
    
    // PASS 2: Pack qh[8] - one int per thread-pair (bytes 64-95)
    // qh[g] = qh_thread_2g | (qh_thread_2g+1 << 16)
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        const int t0 = g * 2;      // thread 0 of pair
        const int t1 = g * 2 + 1;  // thread 1 of pair
        const int elem_base_0 = block_offset * 128 + t0 * 8;
        const int elem_base_1 = block_offset * 128 + t1 * 8;
        
        // Pack 8 high crumbs for thread 0
        uint16_t qh0 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t crumb = q6k_extract_qh(src->qh, elem_base_0 + i);
            qh0 |= (static_cast<uint16_t>(crumb) << (i * 2));
        }
        
        // Pack 8 high crumbs for thread 1
        uint16_t qh1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint8_t crumb = q6k_extract_qh(src->qh, elem_base_1 + i);
            qh1 |= (static_cast<uint16_t>(crumb) << (i * 2));
        }
        
        // Pack both into one int: qh0 in low 16, qh1 in high 16
        out_blk.qh[g] = static_cast<int>(qh0) | (static_cast<int>(qh1) << 16);
    }
    
    // PASS 3: Pack scales[8] - one half per thread-pair (bytes 96-111)
    #pragma unroll
    for (int s = 0; s < 8; s++) {
        const int scale_idx = block_offset * 8 + s;
        const float scale = d * float(src->scales[scale_idx]);
        out_blk.scales[s] = __float2half(scale);
    }

    const int dst_idx = col128 * nrows + row;
    reinterpret_cast<block_c_q6_K_k128*>(dst_data)[dst_idx].copy_from(out_blk);
}

// =============================================================================
// UTILITY: GET REPACKED SIZE
// =============================================================================
// K/128 MODE: Returns the size in bytes of the repacked tensor with embedded scales

/// Get the size of repacked weights for a given format
/// Returns: size in bytes, or -1 if format not supported
__host__ __device__ inline int64_t get_repacked_size(
    int nrows,
    int ncols,
    int qtype
) {
    // K/128 formats: [K/128, N, block_bytes]
    // All output 128 elements per block
    const int block_bytes = qtype_output_block_size(qtype);
    if (block_bytes < 0) return -1;
    return (int64_t)nrows * (ncols / 128) * block_bytes;
}

/// Get the size of input weights before repacking
/// Returns: size in bytes, or -1 if format not supported
__host__ __device__ inline int64_t get_input_size(
    int nrows,
    int ncols,
    int qtype
) {
    const int block_bytes = qtype_input_block_size(qtype);
    const int block_elems = qtype_input_block_elems(qtype);
    if (block_bytes < 0 || block_elems <= 0) return -1;
    return (int64_t)nrows * (ncols / block_elems) * block_bytes;
}

// =============================================================================
// AWQ REPACKING
// =============================================================================
// AWQ format: 4-bit asymmetric with per-group scale and zero point
// Dequant formula: w = scale * (q - zero)
//
// Input formats (from HuggingFace AWQ):
// - qweight: [K/8, N] int32 packed 4-bit (8 values per int)
// - qzeros:  [K/group_size, N/8] int32 packed 4-bit zeros
// - scales:  [K/group_size, N] float16 scales
//
// We support two entry points:
// 1. repack_q_awq_impl: Takes pre-packed linear buffer
// 2. repack_q_awq_hf_impl: Takes separate qweight/qzeros/scales tensors (HF format)

/// AWQ input block structure (HuggingFace format)
/// K elements packed as 4-bit with separate scale/zero arrays
typedef struct {
    uint8_t qs[64];    // 128 × 4-bit = 64 bytes
    half scale;        // per-128 scale (for g128)
    half zero;         // per-128 zero point
} block_awq_input_t;

/// Repack AWQ weights (group size 128) to K/128 format
/// 
/// This is the simpler case where group size matches K/128 block size.
/// Each output block has exactly one scale/zero pair.
///
/// Input format (BlockQAWQ from Rust):
///   For each K/128 block: [qs (64B)] [scale (2B)] [zero (2B)] [pad (12B)]
///   Total: 80 bytes per block
///
/// Output: block_c_q_awq_k128[K/128, N]
__device__ void repack_q_awq_impl(
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

    // Input: row-major BlockQAWQ from Rust (80 bytes per block)
    constexpr int INPUT_BLOCK_SIZE = 80;  // 64B qs + 2B scale + 2B zero + 12B padding
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_data);
    const int src_idx = row * blocks_per_row + col128;
    const uint8_t* src_block = src + src_idx * INPUT_BLOCK_SIZE;

    // Output: [K/128, N] column-major
    block_c_q_awq_k128* dst = reinterpret_cast<block_c_q_awq_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q_awq_k128 out_blk;

    // Pack 128 elements (64 bytes of 4-bit nibbles) into 16 ints with LOP3-ready layout
    // Source has standard nibble packing: byte[i] = (elem[2i+1] << 4) | elem[2i]
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = t * 8;
        const int byte_base = elem_base / 2;  // 4 bytes for 8 elements
        
        // Extract 8 nibbles
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint8_t byte_val = src_block[byte_base + i];
            n[i*2]   = byte_val & 0x0F;
            n[i*2+1] = byte_val >> 4;
        }
        
        // Pack with LOP3-ready layout
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        out_blk.qs[t] = qs_val;
    }

    // Copy scale and zero
    out_blk.scale = load_half_unaligned(src_block + 64);
    out_blk.zero = load_half_unaligned(src_block + 66);
    out_blk._pad[0] = 0;
    out_blk._pad[1] = 0;

    dst[dst_idx].copy_from(out_blk);
}

/// Repack AWQ weights (group size 64) to K/128 format
///
/// Each K/128 output block contains 2 groups, each with its own scale/zero.
///
/// Input format (BlockQAWQG64 from Rust, 80 bytes per block):
///   [qs (64B)] [scales[0] (2B)] [scales[1] (2B)] [zeros[0] (2B)] [zeros[1] (2B)] [pad (4B)]
///
/// Output: block_c_q_awq_g64_k128[K/128, N]
__device__ void repack_q_awq_g64_impl(
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

    // Input: row-major BlockQAWQG64 from Rust (80 bytes per block)
    constexpr int INPUT_BLOCK_SIZE = 80;  // 64B qs + 4B scales + 4B zeros + 4B padding
    const uint8_t* src = reinterpret_cast<const uint8_t*>(src_data);
    const int src_idx = row * blocks_per_row + col128;
    const uint8_t* src_block = src + src_idx * INPUT_BLOCK_SIZE;
    
    // Output: [K/128, N] column-major
    block_c_q_awq_g64_k128* dst = reinterpret_cast<block_c_q_awq_g64_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q_awq_g64_k128 out_blk;

    // Process all 16 qs ints (both groups interleaved in Rust: qs[0-7]=g0, qs[8-15]=g1)
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int byte_base = t * 4;
        
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint8_t byte_val = src_block[byte_base + i];
            n[i*2]   = byte_val & 0x0F;
            n[i*2+1] = byte_val >> 4;
        }
        
        const int qs_val = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
        out_blk.qs[t] = qs_val;
    }
    
    // Load scales and zeros from Rust struct layout:
    // Offset 64: scales[0], scales[1]
    // Offset 68: zeros[0], zeros[1]
    out_blk.scales[0] = load_half_unaligned(src_block + 64);
    out_blk.scales[1] = load_half_unaligned(src_block + 66);
    out_blk.zeros[0] = load_half_unaligned(src_block + 68);
    out_blk.zeros[1] = load_half_unaligned(src_block + 70);
    out_blk._pad = 0;

    dst[dst_idx].copy_from(out_blk);
}

/// Repack AWQ from HuggingFace separate tensor format
///
/// HuggingFace AWQ stores:
/// - qweight: [K/8, N] int32 with 8 packed 4-bit weights per int
/// - qzeros:  [K/group_size, N/8] int32 with 8 packed 4-bit zeros per int  
/// - scales:  [K/group_size, N] float16 scales
///
/// This function handles the common case with group_size=128.
__device__ void repack_q_awq_hf_impl(
    const int32_t* __restrict__ qweight,   // [K/8, N] packed weights
    const int32_t* __restrict__ qzeros,    // [K/128, N/8] packed zeros (g128)
    const half* __restrict__ scales,       // [K/128, N] scales
    void* __restrict__ dst_data,
    int nrows,    // N (output features)
    int ncols     // K (input features)
) {
    const int blocks_per_row = ncols / 128;
    const int total_blocks = nrows * blocks_per_row;

    const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= total_blocks) return;

    const int row = block_idx / blocks_per_row;      // N dimension
    const int col128 = block_idx % blocks_per_row;   // K/128 block

    // Output: [K/128, N] column-major
    block_c_q_awq_k128* dst = reinterpret_cast<block_c_q_awq_k128*>(dst_data);
    const int dst_idx = col128 * nrows + row;

    block_c_q_awq_k128 out_blk;

    // Load scale and zero for this group
    // scales layout: [K/128, N] row-major -> index = col128 * nrows + row... 
    // Actually HF is usually [K/g, N] row-major, so scales[col128, row]
    out_blk.scale = scales[col128 * nrows + row];
    
    // qzeros layout: [K/128, N/8] with 8 zeros packed per int
    // Zero for row is at qzeros[col128, row/8], nibble (row % 8)
    const int zero_int_idx = col128 * (nrows / 8) + (row / 8);
    const int zero_nibble = row % 8;
    const int zero_packed = qzeros[zero_int_idx];
    const uint8_t zero_val = (zero_packed >> (zero_nibble * 4)) & 0xF;
    out_blk.zero = __int2half_rn(zero_val);

    // Extract 128 weights from qweight
    // qweight layout: [K/8, N] with 8 weights packed per int (row-major in N)
    // For K position k at row n: qweight[k/8, n], nibble (k % 8)
    const int k_base = col128 * 128;  // Starting K position
    
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        const int elem_base = k_base + t * 8;
        
        uint8_t n[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int k = elem_base + i;
            const int w_int_idx = (k / 8) * nrows + row;
            const int w_nibble = k % 8;
            const int w_packed = qweight[w_int_idx];
            n[i] = (w_packed >> (w_nibble * 4)) & 0xF;
        }
        
        out_blk.qs[t] = pack_nibbles_lop3_ready(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7]);
    }

    out_blk._pad[0] = 0;
    out_blk._pad[1] = 0;

    dst[dst_idx].copy_from(out_blk);
}
