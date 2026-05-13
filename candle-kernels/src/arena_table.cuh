#pragma once

#include <stdint.h>

// ============================================================================
// ARENA TABLE STRUCTURE AND ACCESS HELPERS
// ============================================================================
// 
// Shared header for arena table structure used by paged attention kernels.
// The arena table contains one entry per arena with K/V pointers and metadata.
// Using a proper struct provides type-safe access instead of byte offset indexing.
//
// Metadata encoding: (k_format_tag << 16) | (v_format_tag << 8) | location
//
// Format tags:
//   0=F32, 1=F16, 2=BF16, 3=F8E4M3, 10=Q4_0, 11=Q8_0, 255=Invalid
// Location:
//   0=GPU, 1=CPU
// ============================================================================

/// Arena table entry structure - represents one arena's metadata
/// Layout matches the Rust ArenaTable row layout exactly: (num_arenas, 3) of i64
struct ArenaTableEntry {
    int64_t k_ptr;      // Pointer to K cache arena (as int64_t)
    int64_t v_ptr;      // Pointer to V cache arena (as int64_t)
    int64_t metadata;   // (k_format_tag << 16) | (v_format_tag << 8) | location
};

// Compile-time assertion to ensure struct is packed correctly
static_assert(sizeof(ArenaTableEntry) == 24, "ArenaTableEntry must be 24 bytes (3 x int64_t)");

// ============================================================================
// PER-HEAD TABLE STRUCTURE AND ACCESS HELPERS
// ============================================================================
//
// Extended table with per-head resolution: one entry per (arena, kv_head).
// Layout: entries[arena_idx * n_kv_head + head_idx], GPU shape (num_arenas * n_kv_head, 7) of i64.
// For uniform arenas all n_kv_head entries share the same ptr but have different byte_offsets.
// For PerHeadQuantized arenas each entry has its own ptr and potentially different format.
//
// Columns: [k_ptr, v_ptr, k_byte_offset, v_byte_offset, k_chunk_byte_stride, v_chunk_byte_stride, metadata]
// Metadata encoding same as ArenaTableEntry: (k_format_tag << 16) | (v_format_tag << 8) | location
// ============================================================================

/// Per-head table entry — one row per (arena, kv_head).
struct PerHeadTableEntry {
    int64_t k_ptr;                // Base pointer to K allocation
    int64_t v_ptr;                // Base pointer to V allocation
    int64_t k_byte_offset;        // Byte offset from k_ptr to this head's data
    int64_t v_byte_offset;        // Byte offset from v_ptr to this head's data
    int64_t k_chunk_byte_stride;  // Bytes between consecutive chunks for this head
    int64_t v_chunk_byte_stride;  // Bytes between consecutive chunks for this head
    int64_t metadata;             // (k_format_tag << 16) | (v_format_tag << 8) | location
    int64_t k_outer_scale_bits;   // f32 outer scale for K, stored as bit-cast i64
    int64_t v_outer_scale_bits;   // f32 outer scale for V, stored as bit-cast i64
};

static_assert(sizeof(PerHeadTableEntry) == 72, "PerHeadTableEntry must be 72 bytes (9 x int64_t)");

// Palette4 constants (must appear before Palette4PerHeadEntry struct)
constexpr int N_PALETTE = 4;
constexpr int GIDS_PER_HEAD = N_PALETTE * 2;  // = 8

// ============================================================================
// PALETTE4 PER-HEAD TABLE - 36-column rows (4 × PerHeadTableEntry)
// ============================================================================
// Each row in the per_head_table tensor corresponds to one (arena_idx, kv_head)
// pair and stores N_PALETTE sub-entries.  For the initial implementation all
// N_PALETTE sub-entries are IDENTICAL (same arena, same byte stride, same fmt).
// Future cross-palette quantization can differentiate sub-entries.
//
// Table indexed by: pal0_arena_idx * n_kv_head + head_idx
// ============================================================================
struct Palette4PerHeadEntry {
    PerHeadTableEntry palette[N_PALETTE];
};
static_assert(sizeof(Palette4PerHeadEntry) == 288,
              "Palette4PerHeadEntry must be 288 bytes (4 × 72)");

/// Look up the Palette4 per-head entry for (pal0_arena_idx, kv_head).
__device__ __forceinline__ Palette4PerHeadEntry palette4_per_head_lookup(
    const Palette4PerHeadEntry* __restrict__ table,
    int arena_idx,
    int kv_head_idx,
    int n_kv_head
) {
    return table[arena_idx * n_kv_head + kv_head_idx];
}

/// Extract the PerHeadTableEntry sub-entry for a specific palette.
__device__ __forceinline__ PerHeadTableEntry palette4_sub_entry(
    const Palette4PerHeadEntry& e,
    int palette
) {
    return e.palette[palette];
}

/// Backward-compat: extract palette-0 sub-entry.
__device__ __forceinline__ PerHeadTableEntry palette4_pal0(
    const Palette4PerHeadEntry& e
) {
    return e.palette[0];
}

/// Look up the per-head entry for a given arena and kv_head (Palette4 table, returns palette-0).
/// This backward-compat overload makes existing per_head_lookup(table, ...) calls work when
/// 'table' now points to Palette4PerHeadEntry rows (28 cols) rather than old 7-col rows.
__device__ __forceinline__ PerHeadTableEntry per_head_lookup(
    const Palette4PerHeadEntry* __restrict__ table,
    int arena_idx,
    int kv_head_idx,
    int n_kv_head
) {
    return table[arena_idx * n_kv_head + kv_head_idx].palette[0];
}

/// Look up the per-head entry for a given arena and kv_head.
__device__ __forceinline__ PerHeadTableEntry per_head_lookup(
    const PerHeadTableEntry* __restrict__ table,
    int arena_idx,
    int kv_head_idx,
    int n_kv_head
) {
    return table[arena_idx * n_kv_head + kv_head_idx];
}

/// Get pre-resolved K byte pointer (base + offset) for a per-head entry.
__device__ __forceinline__ const char* per_head_k_ptr(const PerHeadTableEntry& e) {
    return reinterpret_cast<const char*>((uintptr_t)e.k_ptr) + e.k_byte_offset;
}

/// Get pre-resolved V byte pointer (base + offset) for a per-head entry.
__device__ __forceinline__ const char* per_head_v_ptr(const PerHeadTableEntry& e) {
    return reinterpret_cast<const char*>((uintptr_t)e.v_ptr) + e.v_byte_offset;
}

/// Get mutable pre-resolved K byte pointer.
__device__ __forceinline__ char* per_head_k_ptr_mut(const PerHeadTableEntry& e) {
    return reinterpret_cast<char*>((uintptr_t)e.k_ptr) + e.k_byte_offset;
}

/// Get mutable pre-resolved V byte pointer.
__device__ __forceinline__ char* per_head_v_ptr_mut(const PerHeadTableEntry& e) {
    return reinterpret_cast<char*>((uintptr_t)e.v_ptr) + e.v_byte_offset;
}

/// Get K format tag from per-head metadata.
__device__ __forceinline__ int per_head_get_k_format(const PerHeadTableEntry& e) {
    return (int)((e.metadata >> 16) & 0xFF);
}

/// Get V format tag from per-head metadata.
__device__ __forceinline__ int per_head_get_v_format(const PerHeadTableEntry& e) {
    return (int)((e.metadata >> 8) & 0xFF);
}

/// Get location from per-head metadata.
__device__ __forceinline__ int per_head_get_location(const PerHeadTableEntry& e) {
    return (int)(e.metadata & 0xFF);
}

/// Get K outer scale from per-head entry.
__device__ __forceinline__ float per_head_get_k_scale(const PerHeadTableEntry& e) {
    return __int_as_float((uint32_t)(e.k_outer_scale_bits & 0xFFFFFFFFu));
}

/// Get V outer scale from per-head entry.
__device__ __forceinline__ float per_head_get_v_scale(const PerHeadTableEntry& e) {
    return __int_as_float((uint32_t)(e.v_outer_scale_bits & 0xFFFFFFFFu));
}

/// Get K arena pointer from arena table, cast to type T
template <typename T>
__device__ __forceinline__ const T* arena_table_k_ptr(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (const T*)(uintptr_t)arena_table[arena_idx].k_ptr;
}

/// Get V arena pointer from arena table, cast to type T
template <typename T>
__device__ __forceinline__ const T* arena_table_v_ptr(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (const T*)(uintptr_t)arena_table[arena_idx].v_ptr;
}

/// Get mutable K arena pointer from arena table, cast to type T
template <typename T>
__device__ __forceinline__ T* arena_table_k_ptr_mut(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (T*)(uintptr_t)arena_table[arena_idx].k_ptr;
}

/// Get mutable V arena pointer from arena table, cast to type T
template <typename T>
__device__ __forceinline__ T* arena_table_v_ptr_mut(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (T*)(uintptr_t)arena_table[arena_idx].v_ptr;
}

/// Get raw K arena byte pointer (for format-agnostic access)
__device__ __forceinline__ const char* arena_table_k_ptr_raw(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (const char*)(uintptr_t)arena_table[arena_idx].k_ptr;
}

/// Get raw V arena byte pointer (for format-agnostic access)
__device__ __forceinline__ const char* arena_table_v_ptr_raw(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (const char*)(uintptr_t)arena_table[arena_idx].v_ptr;
}

/// Get mutable raw K arena byte pointer (for format-agnostic writes)
__device__ __forceinline__ char* arena_table_k_ptr_raw_mut(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (char*)(uintptr_t)arena_table[arena_idx].k_ptr;
}

/// Get mutable raw V arena byte pointer (for format-agnostic writes)
__device__ __forceinline__ char* arena_table_v_ptr_raw_mut(
    const ArenaTableEntry* __restrict__ arena_table,
    int arena_idx
) {
    return (char*)(uintptr_t)arena_table[arena_idx].v_ptr;
}

/// Get K format tag from arena metadata
__host__ __device__ __forceinline__ int arena_get_k_format(const ArenaTableEntry& entry) {
    return (int)((entry.metadata >> 16) & 0xFF);
}

/// Get V format tag from arena metadata
__host__ __device__ __forceinline__ int arena_get_v_format(const ArenaTableEntry& entry) {
    return (int)((entry.metadata >> 8) & 0xFF);
}

/// Get location from arena metadata (0=GPU, 1=CPU)
__host__ __device__ __forceinline__ int arena_get_location(const ArenaTableEntry& entry) {
    return (int)(entry.metadata & 0xFF);
}

// Format tag constants (matching Rust ArenaFormatTag)
namespace ArenaFormat {
    // =========================================================================
    // FLOAT FORMATS (0-15)
    // =========================================================================
    constexpr int F32 = 0;
    constexpr int F16 = 1;
    constexpr int BF16 = 2;
    constexpr int F8E4M3 = 34;
    constexpr int F8E5M2 = 35;
    
    // Simple quants: QUANT_BASE + GGML type index
    // GGML indices: Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9
    constexpr int Q4_0 = 15;
    constexpr int Q4_1 = 16;
    constexpr int Q5_0 = 12;
    constexpr int Q5_1 = 13;
    constexpr int Q8_0 = 7;
    constexpr int Q8_1 = 8;

    // _KS / _K: attention-sink sub-block scaled formats (candle-specific)
    // 4-element sub-block A (attention sinks) + 28-element sub-block B, each with fine scale.
    constexpr int Q4_KS = 18;
    constexpr int Q8_KS = 10;

    // Q2_0 / Q3_0: simple low-bit formats (candle-specific)
    constexpr int Q2_0 = 22;
    constexpr int Q3_0 = 19;

    // R16: Raw F16 with reserved Q-capture space (candle-specific)
    constexpr int R16 = 3;

    // Ultra-low-bit and FP8-scale formats (candle-specific)
    constexpr int Q0 = 33;
    constexpr int Q1_S = 27;
    constexpr int Q2_S = 25;
    constexpr int Q2_A = 26;
    constexpr int Q2_1 = 23;
    constexpr int Q3_1 = 20;
    constexpr int P2 = 4;

    // Sub-bit palette formats (candle-specific, 32-element blocks)
    constexpr int Q0_V  = 28;
    constexpr int Q1_A  = 29;
    constexpr int Q0_X  = 30;
    constexpr int Q0_M2 = 31;
    constexpr int Q0_M4 = 32;

    // WARNING: K-quants and AWQ are NOT SUPPORTED for paged attention!
    // These have 256/128/64-element blocks which don't match CHUNK_SIZE=32.
    // The format codes are defined here only for completeness; using them
    // with paged attention kernels will produce incorrect results.
    // =========================================================================
    
    // K-quants: QUANT_BASE + GGML type index (NOT SUPPORTED - 256-element blocks)
    // GGML indices: Q2K=10, Q3K=11, Q4K=12, Q5K=13, Q6K=14, Q8K=15
    constexpr int Q2K  = 24;
    constexpr int Q3K  = 21;
    constexpr int Q4K  = 17;
    constexpr int Q5K  = 14;
    constexpr int Q6K  = 11;
    constexpr int Q8K  = 9;
    
    // AWQ formats (GGML indices 100+) (NOT SUPPORTED - 128/64-element blocks)
    constexpr int QAWQ     = 5;
    constexpr int QAWQG64  = 6;
    
    // Invalid sentinel
    constexpr int Invalid = 255;
    
    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================
    
    // Convert GGML type index to ArenaFormat
    __device__ __host__ __forceinline__ constexpr int from_ggml_index(int ggml_idx) {
        return ggml_idx == QAWQ || ggml_idx == QAWQG64;
    }
    
    // Check if format is a float type (not quantized)
    __device__ __host__ __forceinline__ constexpr bool is_float(int format) {
        return format == F32 || format == F16 || format == BF16;
    }
    
    // Check if format is quantized
    __device__ __host__ __forceinline__ constexpr bool is_quantized(int format) {
        return is_float(format) == false;
    }
    
    // Check if format is a simple quant (32-element blocks, supported by paged attn)
    __device__ __host__ __forceinline__ constexpr bool is_simple_quant(int format) {
        return (format >= Q4_0 && format <= Q8_1) || format == Q5_0 || format == Q5_1
            || format == Q4_KS || format == Q8_KS
            || format == Q2_0 || format == Q3_0
            || format == Q0 || format == Q1_S || format == Q2_S
            || format == Q2_A || format == Q2_1 || format == Q3_1
            || format == R16
            || format == Q0_V || format == Q1_A || format == Q0_X
            || format == Q0_M2 || format == Q0_M4;
    }
    
    // Check if format is a K-quant (Q2K-Q8K)
    __device__ __host__ __forceinline__ constexpr bool is_k_quant(int format) {
        return format >= Q2K && format <= Q8K;
    }
    
    // Check if format is AWQ
    __device__ __host__ __forceinline__ constexpr bool is_awq(int format) {
        return format == QAWQ || format == QAWQG64;
    }
    
    // Get element size in bytes for float formats (0 for quantized)
    __device__ __host__ __forceinline__ constexpr int float_elem_size(int format) {
        switch (format) {
            case F32: return 4;
            case F16: return 2;
            case BF16: return 2;
            case F8E4M3: return 1;
            case F8E5M2: return 1;
            default: return 0;  // Quantized or invalid
        }
    }
    
    // Check if format is supported for paged attention (32-element blocks only)
    __device__ __host__ __forceinline__ constexpr bool is_supported(int format) {
        // Float formats: all supported
        if (is_float(format)) return true;
        // Simple quants (32-element blocks): supported
        if (is_simple_quant(format)) return true;
        // K-quants (256-element) and AWQ (128/64-element): NOT supported
        return false;
    }
    
    // Get block size - always 32 for supported formats
    // Use block_size_checked<FORMAT>() for compile-time validation
    __device__ __host__ __forceinline__ constexpr int block_size(int /*format*/) {
        return 32;  // All supported formats use 32-element blocks
    }
    
    // Compile-time checked block size - static_assert on unsupported formats
    template <int FORMAT>
    __device__ __host__ __forceinline__ constexpr int block_size_checked() {
        static_assert(is_supported(FORMAT), 
            "Unsupported format for paged attention! Only 32-element block formats allowed "
            "(F32, F16, BF16, F8E4M3, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1). "
            "K-quants (256-elem) and AWQ (128/64-elem) are NOT supported.");
        return 32;
    }
}

// Location constants
namespace ArenaLocation {
    constexpr int GPU = 0;
    constexpr int CPU = 1;
}

// ============================================================================
// CHUNK SIZE CONSTANT
// ============================================================================
// 
// chunk_size is now a compile-time constant. This provides:
// 1. Faster division/modulo via bit shifts (32 = 2^5)
// 2. Perfect alignment with GGML quantization blocks (which are 32 elements)
// 3. Enables future per-arena format lookup without warp divergence
//    (all threads in a warp access the same chunk → same arena → same format)
// ============================================================================

// CHUNK_SIZE may already be #defined by blocks.cuh as a preprocessor macro.
// Use a separate constexpr name to avoid the macro collision while keeping the
// inline helpers (chunk_div / chunk_mod) that paged attention relies on.
#ifndef ARENA_CHUNK_SIZE
#define ARENA_CHUNK_SIZE 32
#endif
constexpr int ARENA_CHUNK_SIZE_VAL = ARENA_CHUNK_SIZE;
constexpr int ARENA_CHUNK_SIZE_LOG2 = 5;  // log2(32) for fast division

// Fast division and modulo by CHUNK_SIZE using bit operations
__device__ __host__ __forceinline__ constexpr int chunk_div(int pos) {
    return pos >> ARENA_CHUNK_SIZE_LOG2;
}

__device__ __host__ __forceinline__ constexpr int chunk_mod(int pos) {
    return pos & (ARENA_CHUNK_SIZE_VAL - 1);
}

// ============================================================================
// CHUNK META ENTRY
// ============================================================================
//
// Per-block metadata packed as 4 × uint32_t (16 bytes, 16-byte aligned).
// AoS layout: chunk_meta[batch * max_blocks + blk]
//
// Mirrors the Rust `ChunkMeta` struct in candle-nn::kv_cache::chunked::types.
// ============================================================================

/// Per-block metadata entry for paged attention kernels.
///
/// 16 bytes, 16-byte aligned.  Carries block-level usage/offset and RoPE
/// position.  Per-head K/V GIDs are passed separately via the head_gids
/// tensor (see head_gid_k / head_gid_v helpers above).
///
/// Stored as an AoS buffer: `chunk_meta[batch * max_blocks + blk]`.
/// The buffer is built by `ChunkedKvBacking::chunk_meta_row()` on the host
/// and transferred to the device as a flat `uint32_t` tensor.
struct alignas(16) ChunkMeta {
    uint32_t block_usage;    ///< Packed: [15:0] = usage (valid token count), [31:16] = offset (skip from chunk start)
    uint32_t rope_pos_u32;   ///< Absolute RoPE base position for first token (int32_t bit-cast)
    uint32_t _pad0;          ///< Reserved padding to 16 bytes
    uint32_t _pad1;          ///< Reserved padding to 16 bytes

    /// Valid token count in this block [0..32] (low 16 bits of block_usage).
    __device__ __forceinline__ uint32_t usage() const {
        return __ldg(&block_usage) & 0xFFFFu;
    }

    /// Skip-offset from start of physical chunk where valid data begins (high 16 bits).
    __device__ __forceinline__ uint32_t offset() const {
        return __ldg(&block_usage) >> 16;
    }

    /// Absolute RoPE base position for the first valid token in this block (signed).
    __device__ __forceinline__ int32_t rope_base() const {
        return (int32_t)__ldg(&rope_pos_u32);
    }
};
static_assert(sizeof(ChunkMeta) == 16, "ChunkMeta must be exactly 16 bytes");

// ============================================================================
// PALETTE4 CONSTANTS AND PER-HEAD GID LOOKUP HELPERS
// ============================================================================
// Each KV head is split into N_PALETTE=4 independent sub-arenas (palettes).
// Each palette sub-arena stores HEAD_DIM/N_PALETTE elements per token per chunk.
// GIDS_PER_HEAD = N_PALETTE * 2 = 8 (K and V per palette).
//
// Layout in head_gids buffer, per (batch, block, head):
//   index = head_idx * GIDS_PER_HEAD + palette * 2 + kv
//   (kv=0 for K-side, kv=1 for V-side)
//
// Palette sub-arena chunk byte stride = CHUNK_SIZE * (HEAD_DIM/N_PALETTE) * elem_size.
// (N_PALETTE and GIDS_PER_HEAD are defined earlier, before Palette4PerHeadEntry struct)
// ============================================================================

// ============================================================================
// Per-head GID lookup helpers
//
// The head_gids buffer carries per-head K/V chunk GIDs for every block, in the
// same interleaved layout as the Rust HeadGids struct:
//
//   head_gids[(batch * max_blocks + block) * n_kv_head * GIDS_PER_HEAD
//             + head * GIDS_PER_HEAD + palette * 2]     = K GID for (head, palette)
//   head_gids[... + palette * 2 + 1]                    = V GID for (head, palette)
//
// Callers should pre-compute the per-batch base pointer:
//   const int64_t* hg_batch = head_gids + batch_idx * max_blocks * n_kv_head * GIDS_PER_HEAD;
// then call head_gid_k / head_gid_v (palette-0) or head_gid_k_pal / head_gid_v_pal.
// ============================================================================

/// K-side GID for palette 0 (backward-compat alias).
__device__ __forceinline__ int64_t head_gid_k(
    const int64_t* hg_batch,
    int block_idx,
    int kv_head_idx,
    int n_kv_head
) {
    return __ldg(&hg_batch[block_idx * n_kv_head * GIDS_PER_HEAD + kv_head_idx * GIDS_PER_HEAD]);
}

/// V-side GID for palette 0 (backward-compat alias).
__device__ __forceinline__ int64_t head_gid_v(
    const int64_t* hg_batch,
    int block_idx,
    int kv_head_idx,
    int n_kv_head
) {
    return __ldg(&hg_batch[block_idx * n_kv_head * GIDS_PER_HEAD + kv_head_idx * GIDS_PER_HEAD + 1]);
}

/// K-side GID for a specific palette.
__device__ __forceinline__ int64_t head_gid_k_pal(
    const int64_t* hg_batch,
    int block_idx,
    int kv_head_idx,
    int palette,
    int n_kv_head
) {
    return __ldg(&hg_batch[block_idx * n_kv_head * GIDS_PER_HEAD
                           + kv_head_idx * GIDS_PER_HEAD + palette * 2]);
}

/// V-side GID for a specific palette.
__device__ __forceinline__ int64_t head_gid_v_pal(
    const int64_t* hg_batch,
    int block_idx,
    int kv_head_idx,
    int palette,
    int n_kv_head
) {
    return __ldg(&hg_batch[block_idx * n_kv_head * GIDS_PER_HEAD
                           + kv_head_idx * GIDS_PER_HEAD + palette * 2 + 1]);
}

// ============================================================================
// Palette map GID lookup
//
// Palette GIDs are appended AFTER the head_gids section in the same tensor.
// Offset = batch_size * max_blocks * n_kv_head * GIDS_PER_HEAD (i64 elements).
// Layout: palette_gids[(batch * max_blocks + block) * n_kv_head + head]
// The GID resolves to an arena chunk holding HEAD_DIM/4 bytes of 2-bit palette map.
// ============================================================================

/// Compute pointer to the palette GID section (appended after head_gids).
__device__ __forceinline__ const int64_t* palette_gids_section(
    const int64_t* head_gids,
    int batch_size,
    int max_blocks,
    int n_kv_head
) {
    return head_gids + (int64_t)batch_size * max_blocks * n_kv_head * GIDS_PER_HEAD;
}

/// Look up the palette GID for a specific (batch, block, head).
__device__ __forceinline__ int64_t palette_gid_lookup(
    const int64_t* pal_section,
    int batch_idx,
    int block_idx,
    int kv_head_idx,
    int max_blocks,
    int n_kv_head
) {
    if (!pal_section) return -1;
    return __ldg(&pal_section[((int64_t)batch_idx * max_blocks + block_idx) * n_kv_head + kv_head_idx]);
}

/// Extract 2-bit palette index for dimension d from a palette map byte array.
/// palette_map has HEAD_DIM/4 bytes. Returns 0..3.
__device__ __forceinline__ int palette_map_extract(
    const unsigned char* palette_map,
    int dim
) {
    return (palette_map[dim >> 2] >> ((dim & 3) * 2)) & 0x3;
}
