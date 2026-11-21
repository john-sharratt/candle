// The K/128 block structs for the simple and K-quant formats, each with a size
// check.
//
// Defines the `block_c_*_k128` structs from `block_c_q4_0_k128` through
// `block_c_q3_KO_k128`, each with its `static_assert` size lock, with the
// `gemx_tile_traits` K-tile stride struct and the `QK*` guards they depend on.

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
            __half_raw d0;   // thread 0-3 scale
            __half_raw d1;   // thread 4-7 scale
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            __half2_raw _pad0; // padding for alignment
            __half2_raw _pad1; // padding for alignment
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            __half_raw d2;   // thread 8-11 scale
            __half_raw d3;   // thread 12-15 scale
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
            __half2_raw dm0; // thread 0-3 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            __half2_raw dm1; // thread 4-7 scale/min
            __half2_raw dm2; // thread 8-11 scale/min
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            __half2_raw dm3; // thread 12-15 scale/min
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
            __half_raw d0;   // thread 0-3 scale
            int _spad0;// scale padding
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qh4567;// threads 4-7: high bits
            __half_raw d1;   // thread 4-7 scale
            __half_raw d2;   // thread 8-11 scale
            int qh891011;// threads 8-11: high bits
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            int _spad1;// scale padding
            __half_raw d3;   // thread 12-15 scale
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
            __half2_raw dm0; // thread 0-3 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qh4567;// threads 4-7: high bits
            __half2_raw dm1; // thread 4-7 scale/min
            __half2_raw dm2; // thread 8-11 scale/min
            int qh891011;// threads 8-11: high bits
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            __half2_raw dm3; // thread 12-15 scale/min
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
            __half_raw d0;    // thread 0-3 scale
            __half_raw d1;    // thread 4-7 scale
            __half2_raw _pad0;
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
            __half_raw d2;    // thread 8-11 scale
            __half_raw d3;    // thread 12-15 scale
            __half2_raw _pad1;
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
            __half2_raw dm0;   // thread 0-3 (d, m)
            __half2_raw _pad0;
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
            __half2_raw dm1;   // thread 4-7 (d, m)
            __half2_raw dm2;   // thread 8-11 (d, m)
            int2 qs13;   // thread 13: elems 104-111
            int2 qs14;   // thread 14: elems 112-119
            int2 qs15;   // thread 15: elems 120-127
            __half2_raw dm3;   // thread 12-15 (d, m)
            __half2_raw _pad1;
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
            __half2_raw dm0;       // (d,m) for threads 0-1
            uint16_t qs2;    // thread 2
            uint16_t qs3;    // thread 3
            __half2_raw dm1;       // (d,m) for threads 2-3
            uint16_t qs4;    // thread 4
            uint16_t qs5;    // thread 5
            __half2_raw dm2;       // (d,m) for threads 4-5
            uint16_t qs6;    // thread 6
            uint16_t qs7;    // thread 7
            __half2_raw dm3;       // (d,m) for threads 6-7
            uint16_t qs8;    // thread 8
            uint16_t qs9;    // thread 9
            __half2_raw dm4;       // (d,m) for threads 8-9
            uint16_t qs10;   // thread 10
            uint16_t qs11;   // thread 11
            __half2_raw dm5;       // (d,m) for threads 10-11
            uint16_t qs12;   // thread 12
            uint16_t qs13;   // thread 13
            __half2_raw dm6;       // (d,m) for threads 12-13
            uint16_t qs14;   // thread 14
            uint16_t qs15;   // thread 15
            __half2_raw dm7;       // (d,m) for threads 14-15
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
            __half2_raw dm0;       // (d,m) for threads 0-1 (offset 4, aligned!)
            uint8_t qh1;     // thread 1: high bits (offset 8)
            uint8_t pad1;    // padding (offset 9)
            uint16_t qs1;    // thread 1: low bits (offset 10)

            // Group 1: threads 2-3 (bytes 12-23)
            uint16_t qs2;    // thread 2 (offset 12)
            uint8_t qh2;     // thread 2: high bits (offset 14)
            uint8_t pad2;    // padding (offset 15)
            __half2_raw dm1;       // (d,m) for threads 2-3 (offset 16, aligned!)
            uint8_t qh3;     // thread 3: high bits (offset 20)
            uint8_t pad3;    // padding (offset 21)
            uint16_t qs3;    // thread 3 (offset 22)
            
            // Group 2: threads 4-5 (bytes 24-35)
            uint16_t qs4;    // thread 4 (offset 24)
            uint8_t qh4;     // thread 4: high bits (offset 26)
            uint8_t pad4;    // padding (offset 27)
            __half2_raw dm2;       // (d,m) for threads 4-5 (offset 28, aligned!)
            uint8_t qh5;     // thread 5: high bits (offset 32)
            uint8_t pad5;    // padding (offset 33)
            uint16_t qs5;    // thread 5 (offset 34)
            
            // Group 3: threads 6-7 (bytes 36-47)
            uint16_t qs6;    // thread 6 (offset 36)
            uint8_t qh6;     // thread 6: high bits (offset 38)
            uint8_t pad6;    // padding (offset 39)
            __half2_raw dm3;       // (d,m) for threads 6-7 (offset 40, aligned!)
            uint8_t qh7;     // thread 7: high bits (offset 44)
            uint8_t pad7;    // padding (offset 45)
            uint16_t qs7;    // thread 7 (offset 46)
            
            // Group 4: threads 8-9 (bytes 48-59)
            uint16_t qs8;    // thread 8 (offset 48)
            uint8_t qh8;     // thread 8: high bits (offset 50)
            uint8_t pad8;    // padding (offset 51)
            __half2_raw dm4;       // (d,m) for threads 8-9 (offset 52, aligned!)
            uint8_t qh9;     // thread 9: high bits (offset 56)
            uint8_t pad9;    // padding (offset 57)
            uint16_t qs9;    // thread 9 (offset 58)
            
            // Group 5: threads 10-11 (bytes 60-71)
            uint16_t qs10;   // thread 10 (offset 60)
            uint8_t qh10;    // thread 10: high bits (offset 62)
            uint8_t pad10;   // padding (offset 63)
            __half2_raw dm5;       // (d,m) for threads 10-11 (offset 64, aligned!)
            uint8_t qh11;    // thread 11: high bits (offset 68)
            uint8_t pad11;   // padding (offset 69)
            uint16_t qs11;   // thread 11 (offset 70)
            
            // Group 6: threads 12-13 (bytes 72-83)
            uint16_t qs12;   // thread 12 (offset 72)
            uint8_t qh12;    // thread 12: high bits (offset 74)
            uint8_t pad12;   // padding (offset 75)
            __half2_raw dm6;       // (d,m) for threads 12-13 (offset 76, aligned!)
            uint8_t qh13;    // thread 13: high bits (offset 80)
            uint8_t pad13;   // padding (offset 81)
            uint16_t qs13;   // thread 13 (offset 82)
            
            // Group 7: threads 14-15 (bytes 84-95)
            uint16_t qs14;   // thread 14 (offset 84)
            uint8_t qh14;    // thread 14: high bits (offset 86)
            uint8_t pad14;   // padding (offset 87)
            __half2_raw dm7;       // (d,m) for threads 14-15 (offset 88, aligned!)
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
            __half2_raw dm0; // thread 0-3 scale/min
            __half2_raw dm1; // thread 4-7 scale/min
            int qs4;   // thread 4: elems 32-39
            int qs5;   // thread 5: elems 40-47
            int qs6;   // thread 6: elems 48-55
            int qs7;   // thread 7: elems 56-63
            int qs8;   // thread 8: elems 64-71
            int qs9;   // thread 9: elems 72-79
            int qs10;  // thread 10: elems 80-87
            int qs11;  // thread 11: elems 88-95
            __half2_raw dm2; // thread 8-11 scale/min
            __half2_raw dm3; // thread 12-15 scale/min
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

// Q4_KO K/128: byte-permuted twin of Q4_K. DE-INTERLEAVED q8a128 int8 path: this
// 64-byte block is the QUANT half only — the 16 qs ints (0-63), each sub's four ints
// stored INTERLEAVED as [I0,I2,I1,I3] so a lane's qlo (K[0:16]) and qhi (K[16:32]) are
// adjacent (one int2 LDS.64). The four per-32 (scale,-min) pairs are pulled out into a
// separate 16B scale region at the tensor tail (one dm block per quant block, same
// block index) and read straight from global in the fold — never staged. Keeps the
// cp.async weight stream pure quants. See is_scale_separate + the dm-hoist in kernel.cuh.
typedef struct __align__(16) {
    int qs[16];   // 0-63: [I0,I2,I1,I3] per sub
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 4; i++) d[i] = s[i];
    }
} block_c_q4_KO_k128;
static_assert(sizeof(block_c_q4_KO_k128) == 64, "block_c_q4_KO_k128 must be 64 bytes (quant only; scales separate)");

// MXFP4_KO per-128 quant stand-in — the block_compact<> key type for the native-MXFP4
// per-sub int8 format. Structurally identical to block_c_q4_KO_k128 (64 B of
// nibbles) but a DISTINCT type so `block_compact<block_c_mxfp4>` maps to the MXFP4 k1024
// chunk (not the Q4_KO one). The nibbles are MXFP4 E2M1 codebook indices [0,15], not affine
// [0,15]; the per-sub E8M0 scales ride the k1024 chunk (see block_c_mxfp4_k1024).
// This type is only used as a template key — never streamed itself.
typedef struct __align__(16) {
    int qs[16];   // 0-63: [I0,I2,I1,I3] per sub (same lane interleave as Q4_KO)
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 4; i++) d[i] = s[i];
    }
} block_c_mxfp4_k128;
static_assert(sizeof(block_c_mxfp4_k128) == 64, "block_c_mxfp4_k128 must be 64 bytes (quant-only key type)");

// Q5_K K/128: 16 threads × 8 elements, 4 scale pairs (32-elem groups)
// Same layout as Q5_1 - 4 threads share a scale pair
typedef struct __align__(16) {
    union {
        struct {
            __half2_raw dm0; // thread 0-3 scale/min
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
            __half2_raw dm1; // thread 4-7 scale/min
            __half2_raw dm2; // thread 8-11 scale/min
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
            __half2_raw dm3; // thread 12-15 scale/min
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

// Q5_KO: byte-permuted twin of Q5_K. Same data — 128 low-nibbles, four 5th-bit
// ints, four (scale,-min) pairs — reordered so the qs occupy 0-63, the qh ints
// contiguous (64-79) and the scales grouped at the tail (80-95). Within each sub the
// four qs ints are interleaved [I0,I2,I1,I3] (so the int8 path loads them with one
// int2). A pure permutation for the q8a128 int8 path. 112 bytes / 128 elements.
// DE-INTERLEAVED q8a128 int8 path: this 80-byte block is the QUANT half only
// (qs + qh). The per-32 scales (dm) are pulled out into a separate scale region at
// the tail of the weight tensor (one 16B dm block per quant block, same block index)
// and read straight from global in the fold — never staged. This keeps the cp.async
// weight stream pure quants (the MMA skips the float sectors), drops the old 16B pad,
// and shrinks the weight smem. See is_scale_separate (gemx_dequant.cuh) + the
// dm-hoist in kernel.cuh.
typedef struct __align__(16) {
    int qs[16];   // 0-63:  [I0,I2,I1,I3] per sub
    int qh[4];    // 64-79: 5th-bit ints, one per thread-quad
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 5; i++) d[i] = s[i];
    }
} block_c_q5_KO_k128;
static_assert(sizeof(block_c_q5_KO_k128) == 80, "block_c_q5_KO_k128 must be 80 bytes (quant only; scales separate)");

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

// Q6_KO: DE-INTERLEAVED q8a128 int8 path. The Q6_K compact block is ql (0-63), qh
// (64-95), scales (96-111); this 96-byte block is the QUANT half only (ql + qh). The
// 8 per-16 scales (16B) are pulled out into a separate scale region at the tensor tail
// (one 16B scale block per quant block, same block index, read as 4 half2 in the fold)
// — never staged. Keeps the cp.async weight stream pure quants. See is_scale_separate +
// the dm-hoist in kernel.cuh.
typedef struct __align__(16) {
    int ql[16];      // 0-63
    int qh[8];       // 64-95
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 6; i++) d[i] = s[i];
    }
} block_c_q6_KO_k128;
static_assert(sizeof(block_c_q6_KO_k128) == 96, "block_c_q6_KO_k128 must be 96 bytes (quant only; scales separate)");

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
            __half2_raw d0;    // thread 0-3 scale (d0.x = d, d0.y = unused)
            __half2_raw _pad0;
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
            __half2_raw d1;    // thread 4-7 scale (same value, d1.x = d)
            __half2_raw d2;    // thread 8-11 scale (same value, d2.x = d)
            int2 qs13;   // thread 13: elems 104-111
            int2 qs14;   // thread 14: elems 112-119
            int2 qs15;   // thread 15: elems 120-127
            __half2_raw d3;    // thread 12-15 scale (same value, d3.x = d)
            __half2_raw _pad1;
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

// Q8_KO: DE-INTERLEAVED q8a128 int8 path. The 16 qs int2 made contiguous (0-127); this
// 128-byte block is the QUANT half only. The (replicated) block scale is pulled out into
// a separate 16B scale region at the tensor tail (one per quant block, same block index,
// read as 4 half2 in the fold — Q8 is symmetric so the high half / min is 0). Drops the
// old 16B pad AND keeps the cp.async weight stream pure quants. See is_scale_separate +
// the dm-hoist in kernel.cuh.
typedef struct __align__(16) {
    int2 qs[16];  // 0-127:  thread t's 8 int8
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 8; i++) d[i] = s[i];
    }
} block_c_q8_KO_k128;
static_assert(sizeof(block_c_q8_KO_k128) == 128, "block_c_q8_KO_k128 must be 128 bytes (quant only; scales separate)");

// Q2_KO K/128: the smallest KO twin. One row's 128 values × 2-bit = 32 B (quant only; the
// per-128 (scale,min) rides the k1024 chunk's `dm`). Byte layout mirrors Q6_KO's 2-bit crumb
// region — for `(lane q3, sub)` the uint16 at `q3*8 + sub*2` is `{cr0, cr1}`, where `cr0`
// packs the 4 LOW-half values (K = q3*4 + {0..3}) at bit positions 0,2,4,6 and `cr1` the 4
// HIGH-half values (K = q3*4 + 16 + {0..3}) — but here the crumb IS the whole value (0..3), not
// Q6's high-2-bits. Byte-identical to CPU `quantize_q2_ko`. int2 LDS at `lane*8` pulls all 4
// subs' crumb uint16s.
typedef struct __align__(16) {
    int qs[8];   // 0-31: 4 subs × {cr0,cr1} uint16, at (q3-local) sub*2
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 2; i++) d[i] = s[i];
    }
} block_c_q2_KO_k128;
static_assert(sizeof(block_c_q2_KO_k128) == 32, "block_c_q2_KO_k128 must be 32 bytes (quant only; scales separate)");

// Q3_KO K/128: `Q3_K`'s same-width twin. One row's 128 values × 3-bit = 48 B, spent across the
// two auxiliary planes the other twins already carry and NO `ql` plane: a 32 B **crumb** region
// holding bits 0-1 (byte-identical to Q2_KO's, which is Q6_KO's high-crumb region) and a 16 B
// **hi** region holding bit 2 (byte-identical to Q5_KO's 5th-bit region). Chunk-flat that is
// crumb[0,256) + hi[256,384) + dm[384,416) — see loader/q3_KO.cuh, which is the Q2_KO crumb read
// plus the Q5_KO hi read with the bit landing at 2 instead of 4. Byte-identical to CPU
// `ko_quant::quantize_ko(.., Q3_KO)`.
typedef struct __align__(16) {
    int qs[8];   // 0-31:  crumb — 4 subs × {cr0,cr1} uint16, at (q3-local) sub*2
    int qh[4];   // 32-47: bit-2 ints, one per thread-quad
    template<typename T>
    __device__ __forceinline__ void copy_from(const T& src) {
        const int4* s = reinterpret_cast<const int4*>(&src);
        int4* d = reinterpret_cast<int4*>(this);
        #pragma unroll
        for (int i = 0; i < 3; i++) d[i] = s[i];
    }
} block_c_q3_KO_k128;
static_assert(sizeof(block_c_q3_KO_k128) == 48, "block_c_q3_KO_k128 must be 48 bytes (quant only; scales separate)");
