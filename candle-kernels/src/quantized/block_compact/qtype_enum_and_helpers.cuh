// The `QType` enum with its size constants, the functions that map a `QType` to
// its kernel and block layout, and the short aliases for the K/128 types.
//
// Declares `QType` with its size constants and the `static_assert`s that lock
// its values, the `qtype_*` functions that map a `QType` to its matmul kernel
// index and block sizes, the `BLOCK_SIZE_*` defines, the `block_c_*` aliases
// for the K/128 types, and the `smem_row_stride` and default 4-bit block used
// by `gemx_launch_simple`.

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
    // Kernel-only QTypes past the GgmlDType-aligned range (no GgmlDType mirror).
    // q8a128 is the contiguous q8 *activation* block (block_q8a128, 144B) — the
    // activation-ordered twin of the q8_1 weight block. It is not a stored-weight
    // dtype, so it has no GgmlDType; 36 is the first slot beyond the mirror.
    // Two qtypes for the two matmul modes — SAME block_q8a128, (eventually) different
    // tile layouts: V = mode-1 (Bm=16, decode), X = mode-2 (Bm=32 weight-reuse, prefill).
    QTYPE_Q8A128V = 36,
    QTYPE_Q8A128X = 37,

    // Byte-permuted ("ordered") twins of the K-quant compact blocks — qs made
    // contiguous and the per-sub scales grouped at the tail — for the q8a128 int8
    // matmul path. Weight-only kernel formats produced by an on-GPU permutation of
    // the corresponding K block; they mirror GgmlDType but never hit disk. Values
    // start at 45 because GgmlDType 36-44 are the raw storage dtypes (U8..F64),
    // which this quant-only enum doesn't carry — so KO is the first slot free in
    // BOTH enums, preserving the GgmlDType-as-u32 == QTYPE invariant.
    QTYPE_Q4_KO  = 45,
    QTYPE_Q5_KO  = 46,
    QTYPE_Q6_KO  = 47,
    QTYPE_Q8_KO  = 48,

    // Native OCP MXFP4 storage (mirrors GgmlDType::MXFP4); has no matmul kernel of its own
    // — the routed experts are repacked to the lane-major per-sub twin MXFP4_KO below.
    QTYPE_MXFP4    = 49,
    // Lane-major per-sub MXFP4 for the q8a128 int8 path: one int32 MMA per 32-K sub, each
    // folded with its own E8M0 scale in FP (see loader/mxfp4.cuh + the is_mxfp4_persub
    // branch in kernel.cuh) — exact. Stays 4-bit in storage. First slot past Q8_KO.
    QTYPE_MXFP4_KO = 50,

    // Lane-major per-128 affine KO twin at 2-bit (value 0..3). Smallest KO weight: 32 B of
    // quants per 128-K (vs Q4_KO's 64 B) — the 2-bit crumb region Q6_KO already carries, used
    // here as the whole value. Read by the maintained per-128 int8 fold. First slot past MXFP4_KO.
    QTYPE_Q2_KO  = 51,

    // Lane-major per-128 affine KO twin at 3-bit (value 0..7) — `Q3_K`'s same-width twin. 48 B
    // of quants per 128-K, spent as a 32 B crumb plane (bits 0-1, Q2_KO's region) plus a 16 B
    // hi plane (bit 2, Q5_KO's region) and no `ql` at all. Exists because without it `to_ko`
    // rounded Q3_K *up* to Q4_KO, which under layer streaming is PCIe bytes on every forward.
    // First slot past Q2_KO.
    QTYPE_Q3_KO  = 52,

    QTYPE_COUNT   = 53
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
static_assert(QTYPE_Q8A128V == 36, "QTYPE_Q8A128V must be 36 (first kernel-only QType)");
static_assert(QTYPE_Q8A128X == 37, "QTYPE_Q8A128X must be 37 (q8a128 mode-2)");
static_assert(QTYPE_Q4_KO   == 45, "QTYPE_Q4_KO must be 45");
static_assert(QTYPE_Q5_KO   == 46, "QTYPE_Q5_KO must be 46");
static_assert(QTYPE_Q6_KO   == 47, "QTYPE_Q6_KO must be 47");
static_assert(QTYPE_Q8_KO   == 48, "QTYPE_Q8_KO must be 48");
static_assert(QTYPE_MXFP4    == 49, "QTYPE_MXFP4 must be 49 to match GgmlDType::MXFP4");
static_assert(QTYPE_MXFP4_KO == 50, "QTYPE_MXFP4_KO must be 50");
static_assert(QTYPE_Q2_KO   == 51, "QTYPE_Q2_KO must be 51");
static_assert(QTYPE_Q3_KO   == 52, "QTYPE_Q3_KO must be 52");
static_assert(QTYPE_COUNT   == 53, "QTYPE_COUNT must be 53");

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
        // KO byte-permuted twins: own int8 kernels (different in-block offsets), so
        // own rows past the 14 base formats. FP tables don't have these rows — KO is
        // dispatched only via the q8a128 int8 path (ytype==3).
        case QTYPE_Q4_KO:    return 14;
        case QTYPE_Q5_KO:    return 15;
        case QTYPE_Q6_KO:    return 16;
        case QTYPE_Q8_KO:    return 17;
        case QTYPE_MXFP4_KO: return 18;
        case QTYPE_Q2_KO:    return 19;
        case QTYPE_Q3_KO:    return 20;
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
        case QTYPE_Q4_KO:    return gemx_tile_traits<block_c_q4_KO_k128>::stride;
        case QTYPE_Q5_KO:    return gemx_tile_traits<block_c_q5_KO_k128>::stride;
        case QTYPE_Q6_KO:    return gemx_tile_traits<block_c_q6_KO_k128>::stride;
        case QTYPE_Q8_KO:    return gemx_tile_traits<block_c_q8_KO_k128>::stride;
        case QTYPE_MXFP4_KO: return gemx_tile_traits<block_c_mxfp4_k128>::stride;
        case QTYPE_Q2_KO:    return gemx_tile_traits<block_c_q2_KO_k128>::stride;
        case QTYPE_Q3_KO:    return gemx_tile_traits<block_c_q3_KO_k128>::stride;
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
typedef block_c_q4_KO_k128 block_c_q4_KO;
typedef block_c_q5_K_k128 block_c_q5_K;
typedef block_c_q5_KO_k128 block_c_q5_KO;
typedef block_c_q6_K_k128 block_c_q6_K;
typedef block_c_q6_KO_k128 block_c_q6_KO;
typedef block_c_q8_KO_k128 block_c_q8_KO;
typedef block_c_q2_KO_k128 block_c_q2_KO;
typedef block_c_q3_KO_k128 block_c_q3_KO;
typedef block_c_mxfp4_k128 block_c_mxfp4;
typedef block_c_q8_K_k128 block_c_q8_K;

// Per-row smem stride for the INT8 weight slot = the block byte size (rows packed
// contiguously). A +16B-per-row pad was measured to eliminate the dequant shared-load
// bank conflicts (Q8: 22.1M → 62K) but moved wall-clock by zero — those conflicts were
// latency hidden by TLP, not a throughput bound — while the padded (scattered) cp.async
// destination hurt the single-slot prefetch (Q5 −1.5%→−10%). So: no pad. The trait is
// kept as the single source of truth for the smem row stride.
template <typename T> struct smem_row_stride { static constexpr int value = (int)sizeof(T); };

// Default 4-bit block for gemx_launch_simple
typedef block_c_q4_0_k128 block_c_4bit_default;


// NOTE: Type traits (block_compact<block_q*>) are defined in loaders.cuh
// after both original and compacted types are available.