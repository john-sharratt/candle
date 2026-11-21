// The GEMX format constants, the permutation tables and the packing helpers the
// repacks use.
//
// Holds the `GEMX_TILE_*` and `GEMX_GROUP_SIZE` constants, the permutation
// tables for the 4-bit, 2-bit, 3-bit, Q6_K and Q4_K layouts, and the nibble,
// crumb and LOP3-ready packing helpers.

// =============================================================================
// K/128 OUTPUT: repack to block_c_*_k128 with embedded scales
// =============================================================================

// =============================================================================
// BLOCK SIZE CONSTANTS
// =============================================================================

#ifndef QK4_0
#define QK4_0 32
#endif
#ifndef QK4_1
#define QK4_1 32
#endif
#ifndef QK5_0
#define QK5_0 32
#endif
#ifndef QK5_1
#define QK5_1 32
#endif
#ifndef QK8_0
#define QK8_0 32
#endif
#ifndef QK8_1
#define QK8_1 32
#endif

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#ifndef QK_K
#define QK_K 256
#endif
#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif
#endif
// K-quant repack always reads GGML source blocks with QK_K=256 layout.
#ifdef GGML_QKK_64
#undef QK_K
#undef K_SCALE_SIZE
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

// =============================================================================
// GEMX FORMAT CONSTANTS
// =============================================================================
// GEMX uses specific tile sizes for tensor core operations

// Tile dimensions for GEMX kernel
constexpr int GEMX_TILE_M = 16;   // Output tile rows
constexpr int GEMX_TILE_N = 16;   // Output tile cols  
constexpr int GEMX_TILE_K = 64;   // Reduction dimension per iteration

// Group size for per-group quantization (GEMX expects this)
constexpr int GEMX_GROUP_SIZE = 128;

// =============================================================================
// GEMX PERMUTATION TABLES
// =============================================================================
// Maps: dst_nibble/crumb_idx -> src_nibble/crumb_idx
//
// GEMX's dequant() extracts elements in a specific order for tensor core ops:
// - 4-bit (8 elements/int32): nibbles at bit positions {0,16,4,20,8,24,12,28}
//   which corresponds to element order 0,4,1,5,2,6,3,7 in nibble positions
// - 2-bit (16 elements/int32): similar interleave pattern
//
// These tables remap source layouts to achieve correct element ordering after
// GEMX extraction using shifts {0,16,4,20} for elements 0-3 and {8,24,12,28}
// for elements 4-7.
//
// LOADER REQUIREMENT: After repack, loaders MUST use GEMX extraction shifts,
// NOT linear byte access (get_int_from_uint8). See loader/q4_K.cuh for the
// correct pattern with shifts_lo = {0,16,4,20} and shifts_hi = {8,24,12,28}.
// =============================================================================

// -----------------------------------------------------------------------------
// Q8_0: 32 elements, 8-bit bytes (32 bytes per block)
// -----------------------------------------------------------------------------
// Source layout: linear bytes, element i at byte i
// GEMX needs: interleaved 0,4,1,5,2,6,3,7 per int64 (8 bytes)
// 32 bytes = 4 groups of 8 bytes
//
// The permutation is the INVERSE of [0,4,1,5,2,6,3,7]:
//   dst[0] = src[0], dst[4] = src[1], dst[1] = src[2], dst[5] = src[3], ...
// For gather pattern dst[i] = src[perm[i]]:
//   perm = [0, 2, 4, 6, 1, 3, 5, 7] per 8-byte group
__constant__ uint8_t Q8_PERM[32] = {
      0,   2,   4,   6,   1,   3,   5,   7,   8,  10,  12,  14,   9,  11,  13,  15,
     16,  18,  20,  22,  17,  19,  21,  23,  24,  26,  28,  30,  25,  27,  29,  31,
};

// -----------------------------------------------------------------------------
// 4-bit Simple Formats: Q4_0, Q4_1, Q5_0, Q5_1 (32 elements per block)
// -----------------------------------------------------------------------------
// Source layout: linear nibbles (nibble i = element i)
// GEMX needs: interleaved 0,4,1,5,2,6,3,7 per int32
// This is the inverse of the GEMX extraction pattern:
//   GEMX shift[e] / 4 = nibble position for element e
//   PERM[gemx_pos(e)] = e
__constant__ uint8_t Q4_SIMPLE_PERM[32] = {
      0,   2,   4,   6,   1,   3,   5,   7,   8,  10,  12,  14,   9,  11,  13,  15,
     16,  18,  20,  22,  17,  19,  21,  23,  24,  26,  28,  30,  25,  27,  29,  31,
};

// -----------------------------------------------------------------------------
// K-tile permutation for 4-bit K-tile-major formats (Q4_0, Q4_K, etc.)
// -----------------------------------------------------------------------------
// Maps dst nibble position -> src element within a 16-element K-tile
// Based on GEMX extraction shifts {0,16,4,20,8,24,12,28} which give element order:
//   {0, 4, 1, 5, 2, 6, 3, 7} for first int, {8, 12, 9, 13, 10, 14, 11, 15} for second
__constant__ uint8_t Q4K_TILE_PERM[16] = {
    0, 4, 1, 5, 2, 6, 3, 7,       // First int32: nibbles 0-7  -> elements 0,4,1,5,2,6,3,7
    8, 12, 9, 13, 10, 14, 11, 15  // Second int32: nibbles 8-15 -> elements 8,12,9,13,10,14,11,15
};

// -----------------------------------------------------------------------------
// Q6_K: 256 elements, interleaved between 128-element halves
// -----------------------------------------------------------------------------
// Source layout: Q6_K stores ql[128] (4-bit low) + qh[64] (2-bit high)
//   Elements 0-127 in low half, 128-255 in high half
// GEMX needs: interleave between halves with stride-8 pattern
//   Per int32 (8 nibbles): [base, base+128, base+8, base+136, ...]
//   base = (int32_idx // 4) + (int32_idx % 4) * 32
__constant__ uint8_t Q6K_PERM[256] = {
      0, 128,   8, 136,  16, 144,  24, 152,  32, 160,  40, 168,  48, 176,  56, 184,
     64, 192,  72, 200,  80, 208,  88, 216,  96, 224, 104, 232, 112, 240, 120, 248,
      1, 129,   9, 137,  17, 145,  25, 153,  33, 161,  41, 169,  49, 177,  57, 185,
     65, 193,  73, 201,  81, 209,  89, 217,  97, 225, 105, 233, 113, 241, 121, 249,
      2, 130,  10, 138,  18, 146,  26, 154,  34, 162,  42, 170,  50, 178,  58, 186,
     66, 194,  74, 202,  82, 210,  90, 218,  98, 226, 106, 234, 114, 242, 122, 250,
      3, 131,  11, 139,  19, 147,  27, 155,  35, 163,  43, 171,  51, 179,  59, 187,
     67, 195,  75, 203,  83, 211,  91, 219,  99, 227, 107, 235, 115, 243, 123, 251,
      4, 132,  12, 140,  20, 148,  28, 156,  36, 164,  44, 172,  52, 180,  60, 188,
     68, 196,  76, 204,  84, 212,  92, 220, 100, 228, 108, 236, 116, 244, 124, 252,
      5, 133,  13, 141,  21, 149,  29, 157,  37, 165,  45, 173,  53, 181,  61, 189,
     69, 197,  77, 205,  85, 213,  93, 221, 101, 229, 109, 237, 117, 245, 125, 253,
      6, 134,  14, 142,  22, 150,  30, 158,  38, 166,  46, 174,  54, 182,  62, 190,
     70, 198,  78, 206,  86, 214,  94, 222, 102, 230, 110, 238, 118, 246, 126, 254,
      7, 135,  15, 143,  23, 151,  31, 159,  39, 167,  47, 175,  55, 183,  63, 191,
     71, 199,  79, 207,  87, 215,  95, 223, 103, 231, 111, 239, 119, 247, 127, 255,
};

// -----------------------------------------------------------------------------
// Q2_K: 256 elements, interleaved between 128-element halves (same as Q6_K/Q3_K)
// -----------------------------------------------------------------------------
// Source layout: 256 elements split into two halves (0-127, 128-255)
// GEMX needs: interleave between halves with stride-8 pattern
//   Per int32 (8 crumbs): [base, base+128, base+8, base+136, ...]
//   base = (int32_idx // 4) + (int32_idx % 4) * 32
__constant__ uint16_t Q2K_PERM[256] = {
      0, 128,   8, 136,  16, 144,  24, 152,  32, 160,  40, 168,  48, 176,  56, 184,
     64, 192,  72, 200,  80, 208,  88, 216,  96, 224, 104, 232, 112, 240, 120, 248,
      1, 129,   9, 137,  17, 145,  25, 153,  33, 161,  41, 169,  49, 177,  57, 185,
     65, 193,  73, 201,  81, 209,  89, 217,  97, 225, 105, 233, 113, 241, 121, 249,
      2, 130,  10, 138,  18, 146,  26, 154,  34, 162,  42, 170,  50, 178,  58, 186,
     66, 194,  74, 202,  82, 210,  90, 218,  98, 226, 106, 234, 114, 242, 122, 250,
      3, 131,  11, 139,  19, 147,  27, 155,  35, 163,  43, 171,  51, 179,  59, 187,
     67, 195,  75, 203,  83, 211,  91, 219,  99, 227, 107, 235, 115, 243, 123, 251,
      4, 132,  12, 140,  20, 148,  28, 156,  36, 164,  44, 172,  52, 180,  60, 188,
     68, 196,  76, 204,  84, 212,  92, 220, 100, 228, 108, 236, 116, 244, 124, 252,
      5, 133,  13, 141,  21, 149,  29, 157,  37, 165,  45, 173,  53, 181,  61, 189,
     69, 197,  77, 205,  85, 213,  93, 221, 101, 229, 109, 237, 117, 245, 125, 253,
      6, 134,  14, 142,  22, 150,  30, 158,  38, 166,  46, 174,  54, 182,  62, 190,
     70, 198,  78, 206,  86, 214,  94, 222, 102, 230, 110, 238, 118, 246, 126, 254,
      7, 135,  15, 143,  23, 151,  31, 159,  39, 167,  47, 175,  55, 183,  63, 191,
     71, 199,  79, 207,  87, 215,  95, 223, 103, 231, 111, 239, 119, 247, 127, 255,
};

// -----------------------------------------------------------------------------
// Q3_K: 256 elements, interleaved between 128-element halves (same as Q6_K)
// -----------------------------------------------------------------------------
// Source layout: 256 elements split into two halves (0-127, 128-255)
// GEMX needs: interleave between halves with stride-8 pattern
//   Per int32 (8 crumbs): [base, base+128, base+8, base+136, ...]
//   base = (int32_idx // 4) + (int32_idx % 4) * 32
// hmask (3rd bit) handled separately and copied as-is
__constant__ uint16_t Q3K_PERM[256] = {
      0, 128,   8, 136,  16, 144,  24, 152,  32, 160,  40, 168,  48, 176,  56, 184,
     64, 192,  72, 200,  80, 208,  88, 216,  96, 224, 104, 232, 112, 240, 120, 248,
      1, 129,   9, 137,  17, 145,  25, 153,  33, 161,  41, 169,  49, 177,  57, 185,
     65, 193,  73, 201,  81, 209,  89, 217,  97, 225, 105, 233, 113, 241, 121, 249,
      2, 130,  10, 138,  18, 146,  26, 154,  34, 162,  42, 170,  50, 178,  58, 186,
     66, 194,  74, 202,  82, 210,  90, 218,  98, 226, 106, 234, 114, 242, 122, 250,
      3, 131,  11, 139,  19, 147,  27, 155,  35, 163,  43, 171,  51, 179,  59, 187,
     67, 195,  75, 203,  83, 211,  91, 219,  99, 227, 107, 235, 115, 243, 123, 251,
      4, 132,  12, 140,  20, 148,  28, 156,  36, 164,  44, 172,  52, 180,  60, 188,
     68, 196,  76, 204,  84, 212,  92, 220, 100, 228, 108, 236, 116, 244, 124, 252,
      5, 133,  13, 141,  21, 149,  29, 157,  37, 165,  45, 173,  53, 181,  61, 189,
     69, 197,  77, 205,  85, 213,  93, 221, 101, 229, 109, 237, 117, 245, 125, 253,
      6, 134,  14, 142,  22, 150,  30, 158,  38, 166,  46, 174,  54, 182,  62, 190,
     70, 198,  78, 206,  86, 214,  94, 222, 102, 230, 110, 238, 118, 246, 126, 254,
      7, 135,  15, 143,  23, 151,  31, 159,  39, 167,  47, 175,  55, 183,  63, 191,
     71, 199,  79, 207,  87, 215,  95, 223, 103, 231, 111, 239, 119, 247, 127, 255,
};

// -----------------------------------------------------------------------------
// Q4_K: 256 elements, vLLM GEMX permutation for Q4_K nibble layout
// -----------------------------------------------------------------------------
// This permutation is derived from vLLM's gptq_gemx_repack.cu algorithm,
// adjusted for Q4_K's native nibble layout.
//
// Maps: dst_nibble_idx -> src_nibble_idx (in Q4K's raw byte layout)
//
// Q4K nibble layout (128 bytes = 256 nibbles):
//   - Nibble 2*i     (even): byte i, low nibble  -> element i (for i<64) or element 128+(i-64)
//   - Nibble 2*i+1   (odd):  byte i, high nibble -> element 64+i (for i<64) or element 192+(i-64)
//
// vLLM GEMX element mapping (16x16 tile):
//   First 8 element indices: [0, 128, 8, 136, 16, 144, 24, 152]
//
// This table converts vLLM element indices to Q4K nibble indices:
//   Q4K_PERM[dst] = q4k_element_to_nibble(vllm_elem_perm[dst])
//
// First 8 nibble indices: [0, 128, 16, 144, 32, 160, 48, 176]
//   dst 0 -> elem 0   -> nibble 0   (byte 0 low)
//   dst 1 -> elem 128 -> nibble 128 (byte 64 low)
//   dst 2 -> elem 8   -> nibble 16  (byte 8 low)
//   dst 3 -> elem 136 -> nibble 144 (byte 72 low)
//   ...
__constant__ uint8_t Q4K_PERM[256] = {
      0, 128,  16, 144,  32, 160,  48, 176,  64, 192,  80, 208,  96, 224, 112, 240,
      1, 129,  17, 145,  33, 161,  49, 177,  65, 193,  81, 209,  97, 225, 113, 241,
      2, 130,  18, 146,  34, 162,  50, 178,  66, 194,  82, 210,  98, 226, 114, 242,
      3, 131,  19, 147,  35, 163,  51, 179,  67, 195,  83, 211,  99, 227, 115, 243,
      4, 132,  20, 148,  36, 164,  52, 180,  68, 196,  84, 212, 100, 228, 116, 244,
      5, 133,  21, 149,  37, 165,  53, 181,  69, 197,  85, 213, 101, 229, 117, 245,
      6, 134,  22, 150,  38, 166,  54, 182,  70, 198,  86, 214, 102, 230, 118, 246,
      7, 135,  23, 151,  39, 167,  55, 183,  71, 199,  87, 215, 103, 231, 119, 247,
      8, 136,  24, 152,  40, 168,  56, 184,  72, 200,  88, 216, 104, 232, 120, 248,
      9, 137,  25, 153,  41, 169,  57, 185,  73, 201,  89, 217, 105, 233, 121, 249,
     10, 138,  26, 154,  42, 170,  58, 186,  74, 202,  90, 218, 106, 234, 122, 250,
     11, 139,  27, 155,  43, 171,  59, 187,  75, 203,  91, 219, 107, 235, 123, 251,
     12, 140,  28, 156,  44, 172,  60, 188,  76, 204,  92, 220, 108, 236, 124, 252,
     13, 141,  29, 157,  45, 173,  61, 189,  77, 205,  93, 221, 109, 237, 125, 253,
     14, 142,  30, 158,  46, 174,  62, 190,  78, 206,  94, 222, 110, 238, 126, 254,
     15, 143,  31, 159,  47, 175,  63, 191,  79, 207,  95, 223, 111, 239, 127, 255,
};

// =============================================================================
// PERMUTATION HELPER FUNCTIONS
// =============================================================================

// Extract a 4-bit nibble from byte array (standard nibble indexing)
// nibble 0 = byte 0 lo, nibble 1 = byte 0 hi, nibble 2 = byte 1 lo, etc.
__device__ __forceinline__ uint8_t extract_nibble(const uint8_t* data, int nib_idx) {
    const int byte_idx = nib_idx >> 1;
    const int is_high = nib_idx & 1;
    const uint8_t byte_val = data[byte_idx];
    return is_high ? (byte_val >> 4) : (byte_val & 0x0F);
}

// Extract a 4-bit nibble from Q4_0 format by element index
// Q4_0 packing: qs[j] lo nibble = element j, qs[j] hi nibble = element j+16
// So: elem 0-15 are lo nibbles of qs[0-15], elem 16-31 are hi nibbles of qs[0-15]
__device__ __forceinline__ uint8_t extract_q4_0_element(const uint8_t* qs, int elem) {
    const int byte_idx = elem & 15;  // elem 0-15 → byte 0-15, elem 16-31 → byte 0-15
    const int is_high = elem >> 4;    // elem < 16 → 0 (lo nibble), elem >= 16 → 1 (hi nibble)
    return is_high ? (qs[byte_idx] >> 4) : (qs[byte_idx] & 0x0F);
}

// Extract a 2-bit crumb from byte array
__device__ __forceinline__ uint8_t extract_crumb(const uint8_t* data, int crumb_idx) {
    const int byte_idx = crumb_idx >> 2;
    const int shift = (crumb_idx & 3) << 1;
    return (data[byte_idx] >> shift) & 0x03;
}

// Pack two 4-bit nibbles into a byte (standard layout)
__device__ __forceinline__ uint8_t pack_nibbles(uint8_t lo, uint8_t hi) {
    return (hi << 4) | (lo & 0x0F);
}

// =============================================================================
// LOP3-READY NIBBLE PACKING
// =============================================================================
// Pack 8 nibbles (n0-n7) into an int32 with a layout optimized for LOP3 extraction.
//
// Standard layout packs nibbles sequentially:
//   byte3:[n7|n6]  byte2:[n5|n4]  byte1:[n3|n2]  byte0:[n1|n0]
// 
// LOP3-ready layout enables extraction with just SHIFT instructions (no PRMT):
//   byte3:[n7|n3]  byte2:[n5|n1]  byte1:[n6|n2]  byte0:[n4|n0]
//
// With LOP3 mask 0x000f000f extracting bits[3:0] and bits[19:16]:
//   v         -> (n0, n1) pair  [bits 0-3 and 16-19]
//   v >> 4    -> (n4, n5) pair  [bits 4-7 and 20-23]
//   v >> 8    -> (n2, n3) pair  [bits 8-11 and 24-27]
//   v >> 12   -> (n6, n7) pair  [bits 12-15 and 28-31]
//
// This eliminates 4 PRMT instructions + 2 AND + 1 SHIFT from the dot_y hot path,
// replacing them with just 3 SHIFT instructions.
// =============================================================================
__device__ __forceinline__ int pack_nibbles_lop3_ready(
    uint8_t n0, uint8_t n1, uint8_t n2, uint8_t n3,
    uint8_t n4, uint8_t n5, uint8_t n6, uint8_t n7
) {
    // Layout: bits[3:0]=n0, bits[7:4]=n4, bits[11:8]=n2, bits[15:12]=n6
    //         bits[19:16]=n1, bits[23:20]=n5, bits[27:24]=n3, bits[31:28]=n7
    return ((n0 & 0xF))        |   // bits 0-3
           ((n4 & 0xF) << 4)   |   // bits 4-7
           ((n2 & 0xF) << 8)   |   // bits 8-11
           ((n6 & 0xF) << 12)  |   // bits 12-15
           ((n1 & 0xF) << 16)  |   // bits 16-19
           ((n5 & 0xF) << 20)  |   // bits 20-23
           ((n3 & 0xF) << 24)  |   // bits 24-27
           ((n7 & 0xF) << 28);     // bits 28-31
}

// Pack four 2-bit crumbs into a byte
__device__ __forceinline__ uint8_t pack_crumbs(uint8_t c0, uint8_t c1, uint8_t c2, uint8_t c3) {
    return (c0 & 0x03) | ((c1 & 0x03) << 2) | ((c2 & 0x03) << 4) | ((c3 & 0x03) << 6);
}
