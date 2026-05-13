#pragma once

// =============================================================================
// GEMX WEIGHT REPACKING KERNELS (K/128)
// =============================================================================
// Reorders quantized weights to GEMX format for tensor core kernels.
// K/128 format: 16 threads × 8 elements per thread = 128 elements per block.
//
// Key transformations:
// 1. Extract scales from GGML superblocks and embed into K/128 output
// 2. Permute quants to thread-major layout (each thread owns contiguous elements)
// 3. Pack bits efficiently for warp-level tensor core consumption
//
// For each Q type (all output K/128 blocks):
// - Q4_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales
// - Q4_1: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 dm pairs
// - Q5_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales + high bits
// - Q5_1: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 dm pairs + high bits
// - Q8_0: 4×32-elem GGML blocks → 1×128-elem K/128 block with 4 scales
// - Q2_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 dm pairs
// - Q3_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 dm pairs
// - Q4_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 4 dm pairs
// - Q5_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 4 dm pairs + high bits
// - Q6_K: 1×256-elem / 2 = 128-elem slice → K/128 block with 8 scales
//
// ⚠️  MEMORY LAYOUT REFERENCE: See block_compact.cuh for K/128 struct definitions.
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "block_compact.cuh"

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
