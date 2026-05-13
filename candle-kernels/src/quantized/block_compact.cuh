#pragma once

// NOTE: Do NOT include impl/common.cuh here - it creates a circular dependency
// because impl/common.cuh -> kernel.cuh -> loaders.cuh -> loader/*.cuh need block_c_* types.
// The block_compact type trait specializations are defined in loaders.cuh instead.

// =============================================================================
// COMPACTED BLOCK STRUCTURES (block_c_*) - SCALES EMBEDDED
// =============================================================================
// These structures represent quantized blocks with scales EMBEDDED at the end
// of each K/128 block for use with batched matmul and GEMX kernels.
// This enables:
//   1. Better memory coalescing (scales accessed in separate pattern)
//   2. Potential tensor core usage with repacked weights
//   3. Reduced memory traffic when scales are cached/reused
//
// LAYOUT MODE:
// - K/64 (suffix _k64): 64 elements per struct, scales embedded at end
//
// Naming: 
//   block_c_<format>_k64 = K/64 layout, scales embedded at end of the block
//
// K/64 LAYOUT (scales embedded at end of each block):
// ┌───────────────────┬─────────────────┬──────────────────────────────┐
// │ Format            │ Weights         │ Embedded Scales (per 64)      │
// ├───────────────────┼─────────────────┼──────────────────────────────┤
// │ Q4_0              │ 64 × 4-bit      │ 1 × half2 (d0, d1)           │
// │ Q4_1              │ 64 × 4-bit      │ 2 × half2 (d0,m0)(d1,m1)     │
// │ Q5_0              │ 64 × 5-bit      │ 1 × half2 (d0, d1)           │
// │ Q5_1              │ 64 × 5-bit      │ 2 × half2 (d0,m0)(d1,m1)     │
// │ Q8_0              │ 64 × 8-bit      │ 1 × half2 (d0, d1)           │
// │ Q2_K              │ 64 × 2-bit      │ 4 × half2 (per 16 elements)  │
// │ Q3_K              │ 64 × 3-bit      │ 4 × half2 (per 16 elements)  │
// │ Q4_K              │ 64 × 4-bit      │ 2 × half2 (per 32 elements)  │
// │ Q5_K              │ 64 × 5-bit      │ 2 × half2 (per 32 elements)  │
// │ Q6_K              │ 64 × 6-bit      │ 4 × half (per 16 elements)   │
// └───────────────────┴─────────────────┴──────────────────────────────┘
// =============================================================================

// =============================================================================
// GGML QUANTIZATION BLOCK MEMORY LAYOUTS - AUTHORITATIVE REFERENCE
// =============================================================================
// 
// ⚠️  CRITICAL: These layouts are NOT always intuitive! Each format has unique
//     memory organization inherited from GGML. Always verify against GGML's
//     dequantize_block_* functions in ggml-quants.c before writing extractors.
//
// -----------------------------------------------------------------------------
// Q4_0: 4-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; uint8_t qs[16]; }  // 18 bytes
// 
// qs layout: SEQUENTIAL nibbles
//   qs[i] low nibble  = element 2*i
//   qs[i] high nibble = element 2*i + 1
//
// Dequant: value = d * (nibble - 8)
//
// -----------------------------------------------------------------------------
// Q4_1: 4-bit asymmetric quantization, 32 elements per block  
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t qs[16]; }  // 20 bytes (dm.x=d, dm.y=m)
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
//   qs[i] low nibble  = element 2*i
//   qs[i] high nibble = element 2*i + 1
//
// Dequant: value = d * nibble + m
//
// -----------------------------------------------------------------------------
// Q5_0: 5-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; uint8_t qh[4]; uint8_t qs[16]; }  // 22 bytes
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
// qh layout: SEQUENTIAL bits - bit i of qh[i/8] = high bit of element i
//   qh[i/8] bit (i%8) = 5th bit for element i
//
// Dequant: value = d * ((nibble | (highbit << 4)) - 16)
//
// -----------------------------------------------------------------------------
// Q5_1: 5-bit asymmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t qh[4]; uint8_t qs[16]; }  // 24 bytes
//
// qs layout: SEQUENTIAL nibbles (same as Q4_0)
// qh layout: SEQUENTIAL bits (same as Q5_0)
//
// Dequant: value = d * (nibble | (highbit << 4)) + m
//
// -----------------------------------------------------------------------------
// Q8_0: 8-bit symmetric quantization, 32 elements per block
// -----------------------------------------------------------------------------
// Struct: { half d; int8_t qs[32]; }  // 34 bytes
//
// qs layout: SEQUENTIAL - qs[i] = element i directly
//
// Dequant: value = d * qs[i]
//
// =============================================================================
// K-QUANTS (256 elements per super-block) - MORE COMPLEX LAYOUTS!
// =============================================================================
//
// -----------------------------------------------------------------------------
// Q2_K: 2-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t scales[16]; uint8_t qs[64]; half2 dm; }  // 84 bytes
//
// qs layout: SEQUENTIAL 2-bit values
//   qs[i] contains 4 elements: bits [1:0], [3:2], [5:4], [7:6]
//   Element 4*i+j = (qs[i] >> (2*j)) & 3
//
// scales: 16 × 4-bit scale/min pairs for 16 sub-blocks of 16 elements each
//
// -----------------------------------------------------------------------------
// Q3_K: 3-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; half d; }  // 110 bytes
//
// qs layout: SEQUENTIAL 2-bit low parts
//   Low 2 bits: (qs[i] >> (2*j)) & 3 for element 4*i+j
//
// hmask layout: HIGH BIT INTERLEAVED
//   hmask[i] bit j = high bit for element i + 32*j
//   (NOT sequential like Q5_0!)
//
// scales: 12 bytes encoding 16 × 6-bit scales (complex packing)
//
// -----------------------------------------------------------------------------
// Q4_K: 4-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t scales[12]; uint8_t qs[128]; }  // 144 bytes
//
// qs layout: BLOCK INTERLEAVED (32 low, 32 high pattern)
//   Elements 0-31:    qs[0-31] LOW nibbles
//   Elements 32-63:   qs[0-31] HIGH nibbles  
//   Elements 64-95:   qs[32-63] LOW nibbles
//   Elements 96-127:  qs[32-63] HIGH nibbles
//   Elements 128-159: qs[64-95] LOW nibbles
//   Elements 160-191: qs[64-95] HIGH nibbles
//   Elements 192-223: qs[96-127] LOW nibbles
//   Elements 224-255: qs[96-127] HIGH nibbles
//
// Formula: element e → qs[block_64*32 + (e&31)], nibble = (e&32) ? high : low
//   where block_64 = e / 64
//
// scales: 12 bytes encoding 8 × (6-bit scale, 6-bit min) pairs
//
// -----------------------------------------------------------------------------
// Q5_K: 5-bit quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { half2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }  // 176 bytes
//
// qs layout: BLOCK INTERLEAVED (same as Q4_K!)
//   Elements 0-31:    qs[0-31] LOW nibbles
//   Elements 32-63:   qs[0-31] HIGH nibbles
//   Elements 64-95:   qs[32-63] LOW nibbles
//   Elements 96-127:  qs[32-63] HIGH nibbles
//   Elements 128-159: qs[64-95] LOW nibbles
//   Elements 160-191: qs[64-95] HIGH nibbles
//   Elements 192-223: qs[96-127] LOW nibbles
//   Elements 224-255: qs[96-127] HIGH nibbles
//
// qh layout: TRANSPOSED BIT MATRIX (NOT sequential!)
//   qh[i] bit j = high bit for element (32*j + i)
//   Formula: element e → qh[e % 32] bit (e / 32)
//
// ⚠️  Q5_K qh is DIFFERENT from Q5_0 qh! Q5_0 is sequential, Q5_K is transposed.
//
// scales: 12 bytes encoding 8 × (6-bit scale, 6-bit min) pairs (same as Q4_K)
//
// -----------------------------------------------------------------------------
// Q6_K: 6-bit symmetric quantization, 256 elements per super-block
// -----------------------------------------------------------------------------
// Struct: { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; }  // 210 bytes
//
// ql layout: HALF INTERLEAVED
//   Elements 0-31:   ql[0-31] LOW nibbles
//   Elements 32-63:  ql[32-63] LOW nibbles
//   Elements 64-95:  ql[0-31] HIGH nibbles
//   Elements 96-127: ql[32-63] HIGH nibbles
//   Elements 128-159: ql[64-95] LOW nibbles
//   Elements 160-191: ql[96-127] LOW nibbles
//   Elements 192-223: ql[64-95] HIGH nibbles
//   Elements 224-255: ql[96-127] HIGH nibbles
//
// qh layout: CRUMB INTERLEAVED (2 bits per element)
//   qh[i] contains crumbs for elements i, 32+i, 64+i, 96+i at bits [1:0], [3:2], [5:4], [7:6]
//   Formula: element e → qh[e % 32] bits ((e/32)*2 +: 2)
//
// scales: 16 × int8_t scales for 16 sub-blocks of 16 elements each
//
// =============================================================================
// EXTRACTION HELPER FORMULAS (copy-paste ready)
// =============================================================================
//
// Q4_K/Q5_K qs (block interleaved):
//   block_64 = elem / 64
//   in_block = elem % 64
//   is_high = (in_block >= 32)
//   byte_idx = block_64 * 32 + (in_block & 31)
//   nibble = is_high ? (qs[byte_idx] >> 4) : (qs[byte_idx] & 0xF)
//
// Q5_K qh (transposed bits):
//   byte_idx = elem % 32
//   bit_idx = elem / 32
//   highbit = (qh[byte_idx] >> bit_idx) & 1
//
// Q6_K ql (half interleaved):
//   half_128 = elem / 128
//   in_half = elem % 128
//   is_high = (in_half >= 64)
//   byte_idx = half_128 * 64 + (in_half & 63) - (is_high ? 32 : 0)
//   nibble = is_high ? (ql[byte_idx] >> 4) : (ql[byte_idx] & 0xF)
//
// Q6_K qh (crumb interleaved):
//   byte_idx = elem % 32
//   crumb_idx = elem / 32
//   crumb = (qh[byte_idx] >> (crumb_idx * 2)) & 3
//
// =============================================================================

#include <cuda_fp16.h>
#include <stdint.h>

// =============================================================================
// GEMX K-TILE TRAITS - Compile-time stride and layout information
// =============================================================================
// Use gemx_tile_traits<block_type>::stride to get the K-tile stride in bytes.
// This enables generic GEMX pointer arithmetic across different quant formats.
//
// 32-element K-tiles: Each K-tile contains 32 quantized elements.
// scales_per_ktile: How many scales this K-tile needs (1 for 32-elem groups, 2 for 16-elem groups)
// This design ensures each K-tile is self-contained - no scale sharing across tiles.

template<typename BlockType>
struct gemx_tile_traits {
    // Default: not a GEMX K-tile type
    static constexpr bool is_ktile_major = false;
    static constexpr int stride = 0;              // bytes per 32-elem K-tile
    static constexpr int elements_per_tile = 0;
    static constexpr int bits_per_element = 0;
    static constexpr int scales_per_ktile = 0;    // scales needed per K-tile
};

// Forward declarations for specializations (defined after struct definitions)

// Block sizes (must match original definitions)
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
#ifndef QK_K
#define QK_K 256
#endif
#ifndef K_SCALE_SIZE
#define K_SCALE_SIZE 12
#endif


// =============================================================================
// SIMPLE QUANTS - K/128 BLOCK STRUCTURES (8 threads × 16 elements)
// =============================================================================
// =============================================================================
// SIMPLE QUANTS - K/128 BLOCK STRUCTURES (16 threads × 8 elements)
// =============================================================================
// Each struct contains 128 quantized elements = 16 threads × 8 elements.
// 32-element scale groups: threads 0-3 share d0, 4-7 share d1, 8-11 share d2, 12-15 share d3
// Pattern: [qs0-3][d0][qs4-7][d1][qs8-11][d2][qs12-15][d3] - scales interlaced
// Unions allow access as int, int2, or int4

// Q4_0 K/128: 16 threads × 8 elements, 4 scales (32-elem groups)
// 4-bit: each thread needs int (4B) for 8×4-bit = 32 bits
typedef struct __align__(16) {
    union {
        struct {
            int qs0;   // thread 0: elems 0-7
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            int qs4;   // thread 4: elems 32-39
            half d0;   // thread 0-3 scale
            half d1;   // thread 4-7 scale
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            half2 _pad0; // padding for alignment
            half2 _pad1; // padding for alignment
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            half d2;   // thread 8-11 scale
            half d3;   // thread 12-15 scale
            int qs11;  // thread 11: elems 88-95
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
        };
        int4 pack[5];
        int data[20];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 5; i++) pack[i] = src.pack[i];
    }
} block_c_q4_0_k128;
static_assert(sizeof(block_c_q4_0_k128) == 80, "block_c_q4_0_k128 must be 80 bytes");

// Q4_1 K/128: 16 threads × 8 elements, 4 scale pairs (32-elem groups)
// 4-bit asymmetric: each thread needs int (4B) for 8×4-bit
typedef struct __align__(16) {
    union {
        struct {
            int qs0;   // thread 0: elems 0-7
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            half2 dm0; // thread 0-3 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            half2 dm1; // thread 4-7 scale/min
            half2 dm2; // thread 8-11 scale/min
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            half2 dm3; // thread 12-15 scale/min
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
        };
        int4 pack[5];
        int data[20];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 5; i++) pack[i] = src.pack[i];
    }
} block_c_q4_1_k128;
static_assert(sizeof(block_c_q4_1_k128) == 80, "block_c_q4_1_k128 must be 80 bytes");

// Q5_0 K/128: 16 threads × 8 elements, 4 scales (32-elem groups)
// 5-bit: each thread needs int (4B) low + uint8_t (1B) high
typedef struct __align__(16) {
    union {
        struct {
            int qs0;   // thread 0: elems 0-7 (low 4 bits)
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            int qh0123;// threads 0-3: high bits (packed uint8_t)
            half d0;   // thread 0-3 scale
            int _spad0;// scale padding
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qh4567;// threads 4-7: high bits
            half d1;   // thread 4-7 scale
            half d2;   // thread 8-11 scale
            int qh891011;// threads 8-11: high bits
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            int _spad1;// scale padding
            half d3;   // thread 12-15 scale
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
            int qh12131415;// threads 12-15: high bits
        };
        int4 pack[7];
        int data[28];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 7; i++) pack[i] = src.pack[i];
    }
} block_c_q5_0_k128;
static_assert(sizeof(block_c_q5_0_k128) == 112, "block_c_q5_0_k128 must be 112 bytes");

// Q5_1 K/128: 16 threads × 8 elements, 4 scale pairs (32-elem groups)
// 5-bit asymmetric: each thread needs int (4B) low + uint8_t (1B) high
typedef struct __align__(16) {
    union {
        struct {
            int qs0;   // thread 0: elems 0-7 (low 4 bits)
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            int qh0123;// threads 0-3: high bits (packed uint8_t)
            half2 dm0; // thread 0-3 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qh4567;// threads 4-7: high bits
            half2 dm1; // thread 4-7 scale/min
            half2 dm2; // thread 8-11 scale/min
            int qh891011;// threads 8-11: high bits
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            half2 dm3; // thread 12-15 scale/min
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
            int qh12131415;// threads 12-15: high bits
        };
        int4 pack[7];
        int data[28];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 7; i++) pack[i] = src.pack[i];
    }
} block_c_q5_1_k128;
static_assert(sizeof(block_c_q5_1_k128) == 112, "block_c_q5_1_k128 must be 112 bytes");

// Q8_0 K/128: 16 threads × 8 elements, 4 scales (32-elem groups)
// 8-bit: each thread needs int2 (8B) for 8 elements
typedef struct __align__(16) {
    union {
        struct {
            int2 qs0;   // thread 0: elems 0-7
            int2 qs1;   // thread 1: elems 8-15
            int2 qs2;   // thread 2: elems 16-23
            half d0;    // thread 0-3 scale
            half d1;    // thread 4-7 scale
            half2 _pad0;
            int2 qs3;   // thread 3: elems 24-31
            int2 qs4;   // thread 4: elems 32-39
            int2 qs5;   // thread 5: elems 40-47
            int2 qs6;   // thread 6: elems 48-55
            int2 qs7;   // thread 7: elems 56-63
            int2 qs8;   // thread 8: elems 64-71
            int2 qs9;   // thread 9: elems 72-79
            int2 qs10;  // thread 10: elems 80-87
            int2 qs11;  // thread 11: elems 88-95
            int2 qs12;  // thread 12: elems 96-103
            half d2;    // thread 8-11 scale
            half d3;    // thread 12-15 scale
            half2 _pad1;
            int2 qs13;  // thread 13: elems 104-111
            int2 qs14;  // thread 14: elems 112-119
            int2 qs15;  // thread 15: elems 120-127
        };
        int4 pack[9];
        int data[36];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 9; i++) pack[i] = src.pack[i];
    }
} block_c_q8_0_k128;
static_assert(sizeof(block_c_q8_0_k128) == 144, "block_c_q8_0_k128 must be 144 bytes");

// Q8_1 K/128: 16 threads × 8 elements, 4 dm pairs (32-elem groups)
// Q8_1 is similar to Q8_0 but has dm (d=scale, m=sum) instead of just d
// 8-bit: each thread needs int2 (8B) for 8 elements
// Layout same as Q8_0 but scales are half2 (dm) instead of half (d)
typedef struct __align__(16) {
    union {
        struct {
            int2 qs0;    // thread 0: elems 0-7
            int2 qs1;    // thread 1: elems 8-15
            int2 qs2;    // thread 2: elems 16-23
            half2 dm0;   // thread 0-3 (d, m)
            half2 _pad0;
            int2 qs3;    // thread 3: elems 24-31
            int2 qs4;    // thread 4: elems 32-39
            int2 qs5;    // thread 5: elems 40-47
            int2 qs6;    // thread 6: elems 48-55
            int2 qs7;    // thread 7: elems 56-63
            int2 qs8;    // thread 8: elems 64-71
            int2 qs9;    // thread 9: elems 72-79
            int2 qs10;   // thread 10: elems 80-87
            int2 qs11;   // thread 11: elems 88-95
            int2 qs12;   // thread 12: elems 96-103
            half2 dm1;   // thread 4-7 (d, m)
            half2 dm2;   // thread 8-11 (d, m)
            int2 qs13;   // thread 13: elems 104-111
            int2 qs14;   // thread 14: elems 112-119
            int2 qs15;   // thread 15: elems 120-127
            half2 dm3;   // thread 12-15 (d, m)
            half2 _pad1;
        };
        int4 pack[10];
        int data[40];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 10; i++) pack[i] = src.pack[i];
    }
} block_c_q8_1_k128;
static_assert(sizeof(block_c_q8_1_k128) == 160, "block_c_q8_1_k128 must be 160 bytes");

// =============================================================================
// K-QUANTS - K/128 BLOCK STRUCTURES (16 threads × 8 elements)
// =============================================================================
// Q2_K/Q3_K/Q6_K: 16-element scale groups → 2 threads share a scale
// Q4_K/Q5_K: 32-element scale groups → 4 threads share a scale
// Pattern: scales interlaced after each thread group

// Q2_K K/128: 16 threads × 8 elements, 8 scale pairs (16-elem groups)
// 2-bit: each thread needs uint16_t (2B) for 8×2-bit = 16 bits
// Threads 0-1 share dm0, threads 2-3 share dm1, etc.
typedef struct __align__(16) {
    union {
        struct {
            uint16_t qs0;    // thread 0
            uint16_t qs1;    // thread 1
            half2 dm0;       // (d,m) for threads 0-1
            uint16_t qs2;    // thread 2
            uint16_t qs3;    // thread 3
            half2 dm1;       // (d,m) for threads 2-3
            uint16_t qs4;    // thread 4
            uint16_t qs5;    // thread 5
            half2 dm2;       // (d,m) for threads 4-5
            uint16_t qs6;    // thread 6
            uint16_t qs7;    // thread 7
            half2 dm3;       // (d,m) for threads 6-7
            uint16_t qs8;    // thread 8
            uint16_t qs9;    // thread 9
            half2 dm4;       // (d,m) for threads 8-9
            uint16_t qs10;   // thread 10
            uint16_t qs11;   // thread 11
            half2 dm5;       // (d,m) for threads 10-11
            uint16_t qs12;   // thread 12
            uint16_t qs13;   // thread 13
            half2 dm6;       // (d,m) for threads 12-13
            uint16_t qs14;   // thread 14
            uint16_t qs15;   // thread 15
            half2 dm7;       // (d,m) for threads 14-15
        };
        int4 pack[4];
        int data[16];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 4; i++) pack[i] = src.pack[i];
    }
} block_c_q2_K_k128;
static_assert(sizeof(block_c_q2_K_k128) == 64, "block_c_q2_K_k128 must be 64 bytes");

// Q3_K K/128: 16 threads × 8 elements, 8 scale pairs (16-elem groups)
// 3-bit: each thread needs uint16_t (2B) low + uint8_t (1B) high
// Threads 0-1 share dm0, threads 2-3 share dm1, etc.
// Per group: qs(2B) + qh(1B) + pad(1B) + dm(4B) + qh(1B) + pad(1B) + qs(2B) = 12B
// NOTE: dm requires 4-byte alignment. Padding is explicit here to match compiler
// layout and ensure byte offsets in dequant code are correct. Total: 8×12=96 bytes.
typedef struct __align__(16) {
    union {
        struct {
            // Group 0: threads 0-1 (bytes 0-11)
            uint16_t qs0;    // thread 0: low bits (offset 0)
            uint8_t qh0;     // thread 0: high bits (offset 2)
            uint8_t pad0;    // padding for alignment (offset 3)
            half2 dm0;       // (d,m) for threads 0-1 (offset 4, aligned!)
            uint8_t qh1;     // thread 1: high bits (offset 8)
            uint8_t pad1;    // padding (offset 9)
            uint16_t qs1;    // thread 1: low bits (offset 10)

            // Group 1: threads 2-3 (bytes 12-23)
            uint16_t qs2;    // thread 2 (offset 12)
            uint8_t qh2;     // thread 2: high bits (offset 14)
            uint8_t pad2;    // padding (offset 15)
            half2 dm1;       // (d,m) for threads 2-3 (offset 16, aligned!)
            uint8_t qh3;     // thread 3: high bits (offset 20)
            uint8_t pad3;    // padding (offset 21)
            uint16_t qs3;    // thread 3 (offset 22)
            
            // Group 2: threads 4-5 (bytes 24-35)
            uint16_t qs4;    // thread 4 (offset 24)
            uint8_t qh4;     // thread 4: high bits (offset 26)
            uint8_t pad4;    // padding (offset 27)
            half2 dm2;       // (d,m) for threads 4-5 (offset 28, aligned!)
            uint8_t qh5;     // thread 5: high bits (offset 32)
            uint8_t pad5;    // padding (offset 33)
            uint16_t qs5;    // thread 5 (offset 34)
            
            // Group 3: threads 6-7 (bytes 36-47)
            uint16_t qs6;    // thread 6 (offset 36)
            uint8_t qh6;     // thread 6: high bits (offset 38)
            uint8_t pad6;    // padding (offset 39)
            half2 dm3;       // (d,m) for threads 6-7 (offset 40, aligned!)
            uint8_t qh7;     // thread 7: high bits (offset 44)
            uint8_t pad7;    // padding (offset 45)
            uint16_t qs7;    // thread 7 (offset 46)
            
            // Group 4: threads 8-9 (bytes 48-59)
            uint16_t qs8;    // thread 8 (offset 48)
            uint8_t qh8;     // thread 8: high bits (offset 50)
            uint8_t pad8;    // padding (offset 51)
            half2 dm4;       // (d,m) for threads 8-9 (offset 52, aligned!)
            uint8_t qh9;     // thread 9: high bits (offset 56)
            uint8_t pad9;    // padding (offset 57)
            uint16_t qs9;    // thread 9 (offset 58)
            
            // Group 5: threads 10-11 (bytes 60-71)
            uint16_t qs10;   // thread 10 (offset 60)
            uint8_t qh10;    // thread 10: high bits (offset 62)
            uint8_t pad10;   // padding (offset 63)
            half2 dm5;       // (d,m) for threads 10-11 (offset 64, aligned!)
            uint8_t qh11;    // thread 11: high bits (offset 68)
            uint8_t pad11;   // padding (offset 69)
            uint16_t qs11;   // thread 11 (offset 70)
            
            // Group 6: threads 12-13 (bytes 72-83)
            uint16_t qs12;   // thread 12 (offset 72)
            uint8_t qh12;    // thread 12: high bits (offset 74)
            uint8_t pad12;   // padding (offset 75)
            half2 dm6;       // (d,m) for threads 12-13 (offset 76, aligned!)
            uint8_t qh13;    // thread 13: high bits (offset 80)
            uint8_t pad13;   // padding (offset 81)
            uint16_t qs13;   // thread 13 (offset 82)
            
            // Group 7: threads 14-15 (bytes 84-95)
            uint16_t qs14;   // thread 14 (offset 84)
            uint8_t qh14;    // thread 14: high bits (offset 86)
            uint8_t pad14;   // padding (offset 87)
            half2 dm7;       // (d,m) for threads 14-15 (offset 88, aligned!)
            uint8_t qh15;    // thread 15: high bits (offset 92)
            uint8_t pad15;   // padding (offset 93)
            uint16_t qs15;   // thread 15 (offset 94)
        };
        int4 pack[6];
        int data[24];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 6; i++) pack[i] = src.pack[i];
    }
} block_c_q3_K_k128;
static_assert(sizeof(block_c_q3_K_k128) == 96, "block_c_q3_K_k128 must be 96 bytes");

// Q4_K K/128: 16 threads × 8 elements, 4 scale pairs (32-elem groups)
// Same layout as Q4_1 - 4 threads share a scale pair
typedef struct __align__(16) {
    union {
        struct {
            int qs0;   // thread 0: elems 0-7
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            half2 dm0; // thread 0-3 scale/min
            half2 dm1; // thread 4-7 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            half2 dm2; // thread 8-11 scale/min
            half2 dm3; // thread 12-15 scale/min
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
        };
        int4 pack[5];
        int data[20];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 5; i++) pack[i] = src.pack[i];
    }
} block_c_q4_K_k128;
static_assert(sizeof(block_c_q4_K_k128) == 80, "block_c_q4_K_k128 must be 80 bytes");

// Q5_K K/128: 16 threads × 8 elements, 4 scale pairs (32-elem groups)
// Same layout as Q5_1 - 4 threads share a scale pair
typedef struct __align__(16) {
    union {
        struct {
            half2 dm0; // thread 0-3 scale/min
            int qh0123;// threads 0-3: high bits (packed uint8_t)
            int qs0;   // thread 0: elems 0-7 (low 4 bits)
            int qs1;   // thread 1: elems 8-15
            int qs2;   // thread 2: elems 16-23
            int qs3;   // thread 3: elems 24-31
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qh4567;// threads 4-7: high bits
            half2 dm1; // thread 4-7 scale/min
            half2 dm2; // thread 8-11 scale/min
            int qh891011;// threads 8-11: high bits
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            int qs12;  // thread 12: elems 96-103
            int qs13;  // thread 13: elems 104-111
            int qs14;  // thread 14: elems 112-119
            int qs15;  // thread 15: elems 120-127
            int qh12131415;// threads 12-15: high bits
            half2 dm3; // thread 12-15 scale/min
        };
        int4 pack[7];
        int data[28];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 7; i++) pack[i] = src.pack[i];
    }
} block_c_q5_K_k128;
static_assert(sizeof(block_c_q5_K_k128) == 112, "block_c_q5_K_k128 must be 112 bytes");

// Q6_K K/128: 16 threads × 8 elements, 8 scales (16-elem groups)
// 6-bit: each thread needs int (4B) low + uint16_t (2B) high
// 
// COMPACT 112-BYTE LAYOUT (eliminates 16B padding, better coalescing):
//   Bytes 0-63:   ql[16]    - 16 ints, one per thread (PERFECT coalescing!)
//   Bytes 64-95:  qh[8]     - 8 ints, one per thread-pair (qh_lo | qh_hi << 16)
//   Bytes 96-111: scales[8] - 8 halfs, one per thread-pair
//
// Access patterns:
//   vec_dot (load_part): ql[lane] is perfectly contiguous across all 16 threads!
//   MMA (dequant):       int2 ql load + separate qh + scale loads
//
// Benefits vs 128-byte interlaced:
//   - 12.5% memory reduction (112B vs 128B)
//   - Perfect ql coalescing for vec_dot path
//   - Same extraction math (branchless, LOP3-ready)
typedef struct __align__(16) {
    int ql[16];      // bytes 0-63: one per thread (thread t reads ql[t])
    int qh[8];       // bytes 64-95: one per thread-pair (qh_lo | qh_hi << 16)
    half scales[8];  // bytes 96-111: one per thread-pair
    
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        // Copy 112 bytes = 7 × int4 (optimized vectorized copy)
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 7; i++) d[i] = s[i];
    }
} block_c_q6_K_k128;
static_assert(sizeof(block_c_q6_K_k128) == 112, "block_c_q6_K_k128 must be 112 bytes");

// Q8_K K/128: 16 threads × 8 elements, 1 scale for entire block
// Q8_K is the simplest K-quant: 8-bit quants with a single float scale per 256 elements.
// For K/128, we split the 256-element super-block into 2 blocks of 128 elements.
// Each K/128 block stores the scale (converted to half) embedded at the end.
//
// LAYOUT (144 bytes, same as Q8_0):
//   qs0..qs15: 16 × int2 = 128 int8 values (one int2 per thread)
//   scale: half (shared for all 128 elements)
//
// Dequant formula: value = scale * q8 (where q8 is int8)
typedef struct __align__(16) {
    union {
        struct {
            int2 qs0;    // thread 0: elems 0-7
            int2 qs1;    // thread 1: elems 8-15
            int2 qs2;    // thread 2: elems 16-23
            half2 d0;    // thread 0-3 scale (d0.x = d, d0.y = unused)
            half2 _pad0;
            int2 qs3;    // thread 3: elems 24-31
            int2 qs4;    // thread 4: elems 32-39
            int2 qs5;    // thread 5: elems 40-47
            int2 qs6;    // thread 6: elems 48-55
            int2 qs7;    // thread 7: elems 56-63
            int2 qs8;    // thread 8: elems 64-71
            int2 qs9;    // thread 9: elems 72-79
            int2 qs10;   // thread 10: elems 80-87
            int2 qs11;   // thread 11: elems 88-95
            int2 qs12;   // thread 12: elems 96-103
            half2 d1;    // thread 4-7 scale (same value, d1.x = d)
            half2 d2;    // thread 8-11 scale (same value, d2.x = d)
            int2 qs13;   // thread 13: elems 104-111
            int2 qs14;   // thread 14: elems 112-119
            int2 qs15;   // thread 15: elems 120-127
            half2 d3;    // thread 12-15 scale (same value, d3.x = d)
            half2 _pad1;
        };
        int4 pack[10];
        int data[40];
    };
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        #pragma unroll
        for (int i = 0; i < 10; i++) pack[i] = src.pack[i];
    }
} block_c_q8_K_k128;
static_assert(sizeof(block_c_q8_K_k128) == 160, "block_c_q8_K_k128 must be 160 bytes");

// =============================================================================
// AWQ (ACTIVATION-AWARE WEIGHT QUANTIZATION) - K/128 FORMAT
// =============================================================================
// AWQ uses 4-bit asymmetric quantization with per-group scales and zeros.
// Group size is typically 128 (matching our K/128 block size perfectly).
//
// Dequant formula: w = scale * (q - zero)
// where q is 4-bit unsigned [0,15], zero is 4-bit [0,15], scale is FP16.
//
// Layout: 128 × 4-bit weights + 1 scale (half) + 1 zero (half) per block
// This gives exactly one scale/zero pair per K/128 block.
//
// Thread mapping: 16 threads × 8 elements = 128 elements per block
// Each thread loads 1 int (8 × 4-bit nibbles)

// Q_AWQ K/128: 16 threads × 8 elements, 1 scale/zero pair per block
// 4-bit: 64 bytes weights + 4 bytes scale/zero = 68 bytes (padded to 80 for alignment)
// Thread t loads qs[t] (8 × 4-bit nibbles)
typedef struct __align__(16) {
    uint32_t qs[16];      // 16 threads × 1 int (8 nibbles each) = 128 elements
    half scale;      // scale for entire block
    half zero;       // zero point for entire block
    int _pad[3];     // padding to 80 bytes (16-byte aligned)
    
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 5; i++) d[i] = s[i];
    }
} block_c_q_awq_k128;
static_assert(sizeof(block_c_q_awq_k128) == 80, "block_c_q_awq_k128 must be 80 bytes");

// Q_AWQ_G64: K/128 block with group size 64 (2 groups per block)
// 128 × 4-bit weights + 2 × (scale, zero) = 64 + 8 = 72 bytes (padded to 80)
// Threads 0-7 use scales[0]/zeros[0], threads 8-15 use scales[1]/zeros[1]
typedef struct __align__(16) {
    uint32_t qs[16];        // 16 threads × 1 int (8 nibbles each) = 128 elements
    half scales[2];    // scale per 64-element group
    half zeros[2];     // zero per 64-element group
    int _pad;          // padding to 80 bytes (16-byte aligned)
    
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 5; i++) d[i] = s[i];
    }
} block_c_q_awq_g64_k128;
static_assert(sizeof(block_c_q_awq_g64_k128) == 80, "block_c_q_awq_g64_k128 must be 80 bytes");

// =============================================================================
// Q_INT4 (SIGNED 4-BIT INTEGER) - K/128 FORMAT
// =============================================================================
// =============================================================================
// GEMX K-TILE TRAITS SPECIALIZATIONS (K/128 BLOCKS)
// =============================================================================

template<>
struct gemx_tile_traits<block_c_q2_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 64;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 2;
    static constexpr int scales_per_ktile = 8;    // 1 scale per 16 elements
};

template<>
struct gemx_tile_traits<block_c_q3_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 96;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 3;
    static constexpr int scales_per_ktile = 8;    // 1 scale per 16 elements
};

template<>
struct gemx_tile_traits<block_c_q4_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q5_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 112;            // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 5;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q6_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 112;            // bytes per 128-elem block (compact layout)
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 6;
    static constexpr int scales_per_ktile = 8;    // 1 scale per 16 elements
};

template<>
struct gemx_tile_traits<block_c_q4_0_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q4_1_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q5_0_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 112;            // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 5;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q5_1_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 112;            // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 5;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q8_0_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 144;            // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 8;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q8_1_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 160;            // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 8;
    static constexpr int scales_per_ktile = 4;    // 1 dm pair per 32 elements
};

template<>
struct gemx_tile_traits<block_c_q8_K_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 160;            // bytes per 128-elem block (same as Q8_1)
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 8;
    static constexpr int scales_per_ktile = 1;    // 1 scale per 128 elements (shared from 256-elem super-block)
};

// AWQ (Activation-aware Weight Quantization)
template<>
struct gemx_tile_traits<block_c_q_awq_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 1;    // 1 scale/zero per 128 elements
};

template<>
struct gemx_tile_traits<block_c_q_awq_g64_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // bytes per 128-elem block
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 2;    // 2 scale/zero pairs (group size 64)
};

// =============================================================================
// TYPE ALIASES (short names for convenience)
// =============================================================================
// Primary struct names are block_c_q_*_k128, short aliases omit the _k128 suffix
typedef block_c_q_awq_k128 block_c_q_awq;
typedef block_c_q_awq_g64_k128 block_c_q_awq_g64;

// =============================================================================
// SIZE CONSTANTS
// =============================================================================
// All sizes indexed by qtype for easy lookup. Use QTYPE_* enum values.
//
// GGML qtypes (0-11):  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_1, Q8_K
// K/128 qtypes (12-13): Q_AWQ, Q_AWQ_G64
// =============================================================================

// Qtype enumeration for array indexing
enum QType {
    QTYPE_R16 = 3,
    QTYPE_P2 = 4,
    QTYPE_QAWQ = 5,
    QTYPE_QAWQ_G64 = 6,
    QTYPE_Q8_0 = 7,
    QTYPE_Q8_1 = 8,
    QTYPE_Q8_K = 9,
    QTYPE_Q8_KS = 10,
    QTYPE_Q6_K = 11,
    QTYPE_Q5_0 = 12,
    QTYPE_Q5_1 = 13,
    QTYPE_Q5_K = 14,
    QTYPE_Q4_0 = 15,
    QTYPE_Q4_1 = 16,
    QTYPE_Q4_K = 17,
    QTYPE_Q4_KS = 18,
    QTYPE_Q3_0 = 19,
    QTYPE_Q3_1 = 20,
    QTYPE_Q3_K = 21,
    QTYPE_Q2_0 = 22,
    QTYPE_Q2_1 = 23,
    QTYPE_Q2_K = 24,
    QTYPE_Q2_S = 25,
    QTYPE_Q2_A = 26,
    QTYPE_Q1_S = 27,
    QTYPE_Q0_V = 28,
    QTYPE_Q1_A = 29,
    QTYPE_Q0_X = 30,
    QTYPE_Q0_M2 = 31,
    QTYPE_Q0_M4 = 32,
    QTYPE_Q0 = 33,
    QTYPE_F8E4M3 = 34,
    QTYPE_F8E5M2 = 35,

    QTYPE_COUNT   = 36
};

// =============================================================================
// QType value lock-in
// =============================================================================
// The integer values below MUST be kept in sync with the Rust `GgmlDType`
// (candle-core/src/quantized/mod.rs) and `QType` (candle-kernels/src/quantized/
// api.rs) enums, the `SELECT_FMT_*` macros in select_kv_format.cuh, and the
// `ArenaFormat::*` constants in arena_table.cuh. These static_assert lines
// fail the build the moment anyone reorders the enum, so the rest of the
// dispatch tables (including `qtype_input_block_size`, `qtype_output_block_size`,
// and `run_repack_gemx`) stay safe.
static_assert(QTYPE_R16     == 3,  "QTYPE_R16 must be 3 to match GgmlDType::R16");
static_assert(QTYPE_P2      == 4,  "QTYPE_P2 must be 4");
static_assert(QTYPE_QAWQ    == 5,  "QTYPE_QAWQ must be 5");
static_assert(QTYPE_QAWQ_G64 == 6, "QTYPE_QAWQ_G64 must be 6");
static_assert(QTYPE_Q8_0    == 7,  "QTYPE_Q8_0 must be 7");
static_assert(QTYPE_Q8_1    == 8,  "QTYPE_Q8_1 must be 8");
static_assert(QTYPE_Q8_K    == 9,  "QTYPE_Q8_K must be 9");
static_assert(QTYPE_Q8_KS   == 10, "QTYPE_Q8_KS must be 10");
static_assert(QTYPE_Q6_K    == 11, "QTYPE_Q6_K must be 11");
static_assert(QTYPE_Q5_0    == 12, "QTYPE_Q5_0 must be 12");
static_assert(QTYPE_Q5_1    == 13, "QTYPE_Q5_1 must be 13");
static_assert(QTYPE_Q5_K    == 14, "QTYPE_Q5_K must be 14");
static_assert(QTYPE_Q4_0    == 15, "QTYPE_Q4_0 must be 15");
static_assert(QTYPE_Q4_1    == 16, "QTYPE_Q4_1 must be 16");
static_assert(QTYPE_Q4_K    == 17, "QTYPE_Q4_K must be 17");
static_assert(QTYPE_Q4_KS   == 18, "QTYPE_Q4_KS must be 18");
static_assert(QTYPE_Q3_0    == 19, "QTYPE_Q3_0 must be 19");
static_assert(QTYPE_Q3_1    == 20, "QTYPE_Q3_1 must be 20");
static_assert(QTYPE_Q3_K    == 21, "QTYPE_Q3_K must be 21");
static_assert(QTYPE_Q2_0    == 22, "QTYPE_Q2_0 must be 22");
static_assert(QTYPE_Q2_1    == 23, "QTYPE_Q2_1 must be 23");
static_assert(QTYPE_Q2_K    == 24, "QTYPE_Q2_K must be 24");
static_assert(QTYPE_Q2_S == 25, "QTYPE_Q2_S must be 25");
static_assert(QTYPE_Q2_A == 26, "QTYPE_Q2_A must be 26");
static_assert(QTYPE_Q1_S == 27, "QTYPE_Q1_S must be 27");
static_assert(QTYPE_Q0_V    == 28, "QTYPE_Q0_V must be 28");
static_assert(QTYPE_Q1_A    == 29, "QTYPE_Q1_A must be 29");
static_assert(QTYPE_Q0_X    == 30, "QTYPE_Q0_X must be 30");
static_assert(QTYPE_Q0_M2   == 31, "QTYPE_Q0_M2 must be 31");
static_assert(QTYPE_Q0_M4   == 32, "QTYPE_Q0_M4 must be 32");
static_assert(QTYPE_Q0      == 33, "QTYPE_Q0 must be 33");
static_assert(QTYPE_F8E4M3  == 34, "QTYPE_F8E4M3 must be 34");
static_assert(QTYPE_F8E5M2  == 35, "QTYPE_F8E5M2 must be 35");
static_assert(QTYPE_COUNT   == 36, "QTYPE_COUNT must be 36");

// =============================================================================
// QType -> matmul kernel index
// =============================================================================
// `dispatcher.cu::run_quantized_matmul` owns a 14-entry `kernels[][][]` lookup
// table. Historically the qtype integer itself was used as a direct index into
// this table (so `Q4_0 = 0`, `Q_AWQ_G64 = 13`). Now that every QType value is
// aligned with `GgmlDType` (Q4_0 = 15, Q_AWQ_G64 = 6, etc.), the qtype no
// longer is a contiguous index — this helper maps it back to the 0..13 slot
// the dispatcher's table still uses.
//
// Formats without a matmul kernel return -1; `run_quantized_matmul` treats
// that as "unsupported qtype" and errors out.
__host__ __device__ inline int qtype_to_matmul_kernel_index(int qtype) {
    switch (qtype) {
        case QTYPE_Q4_0:     return 0;
        case QTYPE_Q4_1:     return 1;
        case QTYPE_Q5_0:     return 2;
        case QTYPE_Q5_1:     return 3;
        case QTYPE_Q8_0:     return 4;
        case QTYPE_Q2_K:     return 5;
        case QTYPE_Q3_K:     return 6;
        case QTYPE_Q4_K:     return 7;
        case QTYPE_Q5_K:     return 8;
        case QTYPE_Q6_K:     return 9;
        case QTYPE_Q8_1:     return 10;
        case QTYPE_Q8_K:     return 11;
        case QTYPE_QAWQ:     return 12;
        case QTYPE_QAWQ_G64: return 13;
        default:             return -1;
    }
}

// =============================================================================
// QType -> block layout metadata
// =============================================================================
// Each helper is a `switch(qtype)` keyed on the explicit QTYPE_* symbol and
// returns the value pulled from the existing compile-time sources of truth:
//   - INPUT size / elements: GGML `block_*` struct sizeof / element count.
//   - OUTPUT size: `gemx_tile_traits<block_c_*_k128>::stride`.
// Formats without a GEMX layout (R16, P2, KV-only FP8 types, etc.) return -1.
//
// Using explicit QTYPE_* labels means reordering the enum can never silently
// break these tables — it just surfaces a missing case at compile time via
// -Wswitch if `QType` gets new members.

__host__ __device__ inline int qtype_input_block_size(int qtype) {
    // Sizes match the GGML input `block_q*` struct layouts. These are fixed by
    // the GGML format spec so we encode them as literals here (the GGML block
    // struct typedefs aren't in scope in this header).
    switch (qtype) {
        case QTYPE_Q4_0:     return 18;    // 32 elems: 2B d + 16B qs
        case QTYPE_Q4_1:     return 20;    // 32 elems: 2B d + 2B m + 16B qs
        case QTYPE_Q5_0:     return 22;    // 32 elems: 2B d + 4B qh + 16B qs
        case QTYPE_Q5_1:     return 24;    // 32 elems: 2B d + 2B m + 4B qh + 16B qs
        case QTYPE_Q8_0:     return 34;    // 32 elems: 2B d + 32B qs
        case QTYPE_Q8_1:     return 36;    // 32 elems: 4B dm + 32B qs
        case QTYPE_Q2_K:     return 84;    // 256 elems: 16B scales + 64B qs + 2B d + 2B dmin
        case QTYPE_Q3_K:     return 110;   // 256 elems
        case QTYPE_Q4_K:     return 144;   // 256 elems
        case QTYPE_Q5_K:     return 176;   // 256 elems
        case QTYPE_Q6_K:     return 210;   // 256 elems
        case QTYPE_Q8_K:     return 292;   // 256 elems: 4B d + 256B qs + 32B bsums
        case QTYPE_QAWQ:     return 80;    // 128 elems: 64B qs + scale/zero + padding
        case QTYPE_QAWQ_G64: return 80;    // 128 elems: 64B qs + 2×scale/zero + padding
        default:             return -1;    // Unsupported by GEMX input layout
    }
}

__host__ __device__ inline int qtype_input_block_elems(int qtype) {
    switch (qtype) {
        case QTYPE_Q4_0:
        case QTYPE_Q4_1:
        case QTYPE_Q5_0:
        case QTYPE_Q5_1:
        case QTYPE_Q8_0:
        case QTYPE_Q8_1:     return 32;
        case QTYPE_Q2_K:
        case QTYPE_Q3_K:
        case QTYPE_Q4_K:
        case QTYPE_Q5_K:
        case QTYPE_Q6_K:
        case QTYPE_Q8_K:     return 256;
        case QTYPE_QAWQ:
        case QTYPE_QAWQ_G64: return 128;
        default:             return -1;
    }
}

__host__ __device__ inline int qtype_output_block_size(int qtype) {
    switch (qtype) {
        case QTYPE_Q4_0:     return gemx_tile_traits<block_c_q4_0_k128>::stride;
        case QTYPE_Q4_1:     return gemx_tile_traits<block_c_q4_1_k128>::stride;
        case QTYPE_Q5_0:     return gemx_tile_traits<block_c_q5_0_k128>::stride;
        case QTYPE_Q5_1:     return gemx_tile_traits<block_c_q5_1_k128>::stride;
        case QTYPE_Q8_0:     return gemx_tile_traits<block_c_q8_0_k128>::stride;
        case QTYPE_Q8_1:     return gemx_tile_traits<block_c_q8_1_k128>::stride;
        case QTYPE_Q2_K:     return gemx_tile_traits<block_c_q2_K_k128>::stride;
        case QTYPE_Q3_K:     return gemx_tile_traits<block_c_q3_K_k128>::stride;
        case QTYPE_Q4_K:     return gemx_tile_traits<block_c_q4_K_k128>::stride;
        case QTYPE_Q5_K:     return gemx_tile_traits<block_c_q5_K_k128>::stride;
        case QTYPE_Q6_K:     return gemx_tile_traits<block_c_q6_K_k128>::stride;
        case QTYPE_Q8_K:     return gemx_tile_traits<block_c_q8_K_k128>::stride;
        case QTYPE_QAWQ:     return gemx_tile_traits<block_c_q_awq_k128>::stride;
        case QTYPE_QAWQ_G64: return gemx_tile_traits<block_c_q_awq_g64_k128>::stride;
        default:             return -1;   // Unsupported by GEMX output layout
    }
}

// Backward compat #defines for compile-time constants in repack functions
// These are literal values (not array refs) so they work with constexpr
#define BLOCK_SIZE_Q4_0   18
#define BLOCK_SIZE_Q4_1   20
#define BLOCK_SIZE_Q5_0   22
#define BLOCK_SIZE_Q5_1   24
#define BLOCK_SIZE_Q8_0   34
#define BLOCK_SIZE_Q8_1   36
#define BLOCK_SIZE_Q2_K   84
#define BLOCK_SIZE_Q3_K   110
#define BLOCK_SIZE_Q4_K   144
#define BLOCK_SIZE_Q5_K   176
#define BLOCK_SIZE_Q6_K   210
#define BLOCK_SIZE_Q8_K   292

// =============================================================================
// BACKWARD COMPATIBILITY ALIASES (K/128 types)
// =============================================================================

typedef block_c_q4_0_k128 block_c_q4_0;
typedef block_c_q4_1_k128 block_c_q4_1;
typedef block_c_q5_0_k128 block_c_q5_0;
typedef block_c_q5_1_k128 block_c_q5_1;
typedef block_c_q8_0_k128 block_c_q8_0;
typedef block_c_q8_1_k128 block_c_q8_1;
typedef block_c_q2_K_k128 block_c_q2_K;
typedef block_c_q3_K_k128 block_c_q3_K;
typedef block_c_q4_K_k128 block_c_q4_K;
typedef block_c_q5_K_k128 block_c_q5_K;
typedef block_c_q6_K_k128 block_c_q6_K;
typedef block_c_q8_K_k128 block_c_q8_K;

// Default 4-bit block for gemx_launch_simple
typedef block_c_q4_0_k128 block_c_4bit_default;


// NOTE: Type traits (block_compact<block_q*>) are defined in loaders.cuh
// after both original and compacted types are available.