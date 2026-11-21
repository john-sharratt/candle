// The KO K/1024 chunk blocks, the int8 chunk sizes, the AWQ blocks and the tile
// traits of each format.
//
// Defines the KO K/1024 chunk blocks that the int8 q8a128 path streams, the
// `int8_chunk_bytes` size of a warp's 8-row weight chunk, the AWQ K/128 blocks
// `block_c_q_awq_k128` and `block_c_q_awq_g64_k128`, and the `gemx_tile_traits`
// specializations that describe each format's tile layout.

// =============================================================================
// KO K/1024 CHUNK BLOCKS — the strongly-typed unit the int8 (q8a128) path streams.
// =============================================================================
// One block = one 8-row weight chunk (8 rows × 128 K = 1024 elements) carrying its OWN
// scales inline: the 8 per-row quant sub-blocks followed by 8 `half2` scale pairs — ONE
// (scale, min) PER ROW (i.e. per 128-K), 32 bytes total. (This is per-128, NOT per-sub —
// there is exactly one scale per row, applied after the four k32 sub-MMAs collapse into a
// single int32; see loader/gemx_dequant.cuh for why the fold is per-128.) This replaces
// the old "quant-only k128 block + separate scale region at the tensor tail" layout: now
// the chunk's scales ride the same cp.async as its quants and are read straight from the
// prefetched smem chunk in the fold — no dm-hoist, no second DRAM stream, one unified read
// path. `q[row]` is the existing k128 quant block (bit-layout unchanged); `dm[row]` is its
// per-128 (scale, min).
template <typename QuantRow>
struct block_c_KO_k1024 {
    QuantRow q[8];        // 8 rows of quants (8 * sizeof(QuantRow) bytes); 16-aligned
    half2    dm[8];       // (scale, min) per row — ONE per 128 (per-128 collapse), 32 bytes
};
typedef block_c_KO_k1024<block_c_q4_KO_k128> block_c_q4_KO_k1024;
typedef block_c_KO_k1024<block_c_q5_KO_k128> block_c_q5_KO_k1024;
typedef block_c_KO_k1024<block_c_q6_KO_k128> block_c_q6_KO_k1024;
typedef block_c_KO_k1024<block_c_q8_KO_k128> block_c_q8_KO_k1024;
typedef block_c_KO_k1024<block_c_q2_KO_k128> block_c_q2_KO_k1024;
typedef block_c_KO_k1024<block_c_q3_KO_k128> block_c_q3_KO_k1024;
static_assert(sizeof(block_c_q3_KO_k1024) == 416, "block_c_q3_KO_k1024 must be 416 bytes (256 crumb + 128 hi + 32 dm)");

// MXFP4_KO K/1024 chunk — the native-MXFP4 per-sub int8 format. The 512 B quant
// region is the SAME lane-major layout as block_c_q4_KO_k1024 (byte for (lane, sub, i) at
// lane*16 + sub*4 + i, the uint32 packing K[p] | K[p+16]<<4) — but the nibbles are MXFP4
// E2M1 codebook INDICES [0,15], not affine quants. Each 32-K sub carries its OWN E8M0
// power-of-two scale byte in `e` (e[row*4 + sub]). The dequant is a pure codebook
// expansion; the kernel runs one int32 MMA per sub and folds each with its own scale
// 2^(e_sub-128) in FP (see loader/mxfp4.cuh + the is_mxfp4_persub branch in kernel.cuh)
// — exact, no mantissa truncation, symmetric (E2M1 is centered, no min). `dm[row]` is a
// repack-baked per-row (2^(e_max-128), 0) that the int8 fold does NOT read — it stays in
// the layout so the pack-file format and its fingerprint are unchanged.
struct __align__(16) block_c_mxfp4_k1024 {
    uint8_t ql[512];   // lane-major MXFP4 nibbles (codebook indices): K[p] | K[p+16]<<4
    uint8_t e[32];     // per-sub E8M0 scale bytes: e[row*4 + sub] — read by the per-sub fold
    half2   dm[8];     // repack-baked per-row (2^(e_max-128), 0); unread on the int8 path
};
static_assert(sizeof(block_c_mxfp4_k1024) == 576, "block_c_mxfp4_k1024 must be 576 bytes (512 ql + 32 e + 32 dm)");

// Bytes per warp weight chunk (8 rows) in the int8 path. KO k1024 blocks ARE the 8-row
// chunk (quants + inline scales) → sizeof. Legacy non-KO int8 blocks are per-row → ×8.
template <typename T> struct int8_chunk_bytes { static constexpr int value = 8 * (int)sizeof(T); };
template <> struct int8_chunk_bytes<block_c_q4_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q4_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_q5_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q5_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_q6_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q6_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_q8_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q8_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_q2_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q2_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_q3_KO_k1024> { static constexpr int value = (int)sizeof(block_c_q3_KO_k1024); };
template <> struct int8_chunk_bytes<block_c_mxfp4_k1024> { static constexpr int value = (int)sizeof(block_c_mxfp4_k1024); };

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
    __half_raw scale;      // scale for entire block
    __half_raw zero;       // zero point for entire block
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

// Q4_KO: de-interleaved total footprint = 64B quant (this struct) + 16B scale (separate
// scale region at the tensor tail). The kernel stages only the 64B quant block (sizeof)
// and reads the 16B scale by block index; this stride sizes the whole [quant | scale]
// buffer.
template<>
struct gemx_tile_traits<block_c_q4_KO_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // 64 quant + 16 scale
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 4;
    static constexpr int scales_per_ktile = 4;    // 1 scale per 32 elements
};

// MXFP4_KO: 64B of nibbles (E2M1 codebook indices) + per-sub E8M0 scales. Same 4-bit
// nibble footprint as Q4_KO; the extra per-sub scale bytes ride the k1024 chunk, so the
// per-128 stand-in mirrors the Q4_KO quant footprint for the output-layout stride helper.
template<>
struct gemx_tile_traits<block_c_mxfp4_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 80;             // 64 quant + scale (mirrors Q4_KO)
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

// KO twins — identical metadata to their K counterparts (only the in-block byte
// order differs; Q6_KO is byte-identical to Q6_K).
template<>
struct gemx_tile_traits<block_c_q5_KO_k128> {
    static constexpr bool is_ktile_major = true;
    // De-interleaved total footprint per block: 80B quant (this struct) + 16B scale
    // (separate scale region at the tensor tail). The kernel stages only the 80B quant
    // block (sizeof) and reads the 16B scale by block index; this stride sizes the
    // whole [quant | scale] buffer.
    static constexpr int stride = 96;
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 5;
    static constexpr int scales_per_ktile = 4;
};
template<>
struct gemx_tile_traits<block_c_q6_KO_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 112;            // 96 quant + 16 scale
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 6;
    static constexpr int scales_per_ktile = 8;
};
template<>
struct gemx_tile_traits<block_c_q8_KO_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 144;            // 128 quant + 16 scale (old 16B pad removed)
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 8;
    static constexpr int scales_per_ktile = 4;
};
template<>
struct gemx_tile_traits<block_c_q2_KO_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 48;             // 32 quant + 16 scale
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 2;
    static constexpr int scales_per_ktile = 4;
};
template<>
struct gemx_tile_traits<block_c_q3_KO_k128> {
    static constexpr bool is_ktile_major = true;
    static constexpr int stride = 64;             // 48 quant (32 crumb + 16 hi) + 16 scale
    static constexpr int elements_per_tile = 128;
    static constexpr int bits_per_element = 3;
    static constexpr int scales_per_ktile = 4;
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
