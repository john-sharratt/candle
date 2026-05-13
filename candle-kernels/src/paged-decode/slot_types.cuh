#pragma once
// ============================================================================
// Persistent Slot Buffer Type Accessors
//
// Provides byte-layout accessors for the variable-size structures serialized
// by SlotPool on the Rust side and consumed by the v2 decode kernel.
//
// Byte layouts must match the Rust serialization in slot_state.rs exactly.
//
// SlotState header (24 bytes, fixed):
//   [0..4)   uint32_t n_slices
//   [4..8)   uint32_t write_slice
//   [8..16)  uint64_t slices_ptr       (device pointer into slices tensor)
//   [16..24) uint64_t position_map_ptr (device pointer into per-slot
//                                        position_map: u32[total_tokens]
//                                        where each entry packs
//                                        (slice_idx << 16) | in_blk.
//                                        Replaces chunk_div/chunk_mod
//                                        positional math.)
//
// TokenSlice (8 + n_kv_head * kv_head_byte_size(HD) bytes):
//   [0..2)   uint16_t offset
//   [2..4)   uint16_t len        ← committed on-device after each decode step
//   [4..8)   uint32_t rope
//   [8..)    KvHead head[n_kv_head]
//
// KvHead (HD/2 + 104 bytes for HEAD_DIM HD):
//   [0..HD/4)           uint8_t  k_pal[HD/4]
//   [HD/4..HD/2)        uint8_t  v_pal[HD/4]
//   [HD/2..HD/2+32)     uint64_t k_ptr[4]
//   [HD/2+32..HD/2+64)  uint64_t v_ptr[4]
//   [HD/2+64..HD/2+68)  uint8_t  k_fmt[4]
//   [HD/2+68..HD/2+72)  uint8_t  v_fmt[4]
//   [HD/2+72..HD/2+88)  float    k_scale[4]  (outer scale per K palette, default 1.0;
//                                              encoder multiplies, decoder divides — recovers original magnitude)
//   [HD/2+88..HD/2+104) float    v_scale[4]  (outer scale per V palette, default 1.0; same convention as k_scale)
// ============================================================================

#include <stdint.h>

// ============================================================================
// SlotState header layout
// ============================================================================

struct SlotHeader {
    uint32_t n_slices;
    uint32_t write_slice;
    uint64_t slices_ptr;        // device pointer into per-slot slice data
    uint64_t position_map_ptr;  // device pointer into per-slot
                                // position_map: u32[total_tokens],
                                // entry = (slice_idx << 16) | in_blk.
};
static_assert(sizeof(SlotHeader) == 24, "SlotHeader must be 24 bytes");

// Read the SlotHeader for a given slot index from the headers tensor.
__device__ __forceinline__ const SlotHeader& get_slot_header(const uint8_t* headers, int slot_idx) {
    return *reinterpret_cast<const SlotHeader*>(headers + (int64_t)slot_idx * 24);
}

// Resolve a cum_token position to (slice_idx, in_blk) via the slot's
// position_map.  Replaces `chunk_div(k_pos)` / `chunk_mod(k_pos)` for
// reads of the slot's prefix region.  Caller must guarantee
// `k_pos < total_tokens` (i.e., within the slot's valid prefix).
__device__ __forceinline__ void resolve_pos(
    const SlotHeader& slot_hdr,
    int k_pos,
    int& slice_idx,
    int& in_blk
) {
    uint32_t entry = reinterpret_cast<const uint32_t*>(slot_hdr.position_map_ptr)[k_pos];
    slice_idx = (int)(entry >> 16);
    in_blk    = (int)(entry & 0xFFFF);
}

// ============================================================================
// Byte-size helpers (compile-time for HEAD_DIM, runtime for n_kv_head)
// ============================================================================

// Byte size of one KvHead entry for a given HEAD_DIM template parameter.
template <int HD>
__device__ __host__ constexpr int kv_head_byte_size() {
    return HD / 2 + 104;  // k_pal[HD/4] + v_pal[HD/4] + k_ptr[4]*8 + v_ptr[4]*8 + k_fmt[4] + v_fmt[4] + k_scale[4]*4 + v_scale[4]*4
}

// Byte size of one TokenSlice for a given (HEAD_DIM, n_kv_head).
template <int HD>
__device__ __forceinline__ int token_slice_byte_size(int n_kv_head) {
    return 8 + n_kv_head * kv_head_byte_size<HD>();
}

// ============================================================================
// TokenSlice field accessors
// ============================================================================

// Get a pointer to the start of a specific slice.
template <int HD>
__device__ __forceinline__ const uint8_t* get_slice(uint64_t slices_ptr, int slice_idx, int n_kv_head) {
    return reinterpret_cast<const uint8_t*>(slices_ptr)
        + (int64_t)slice_idx * token_slice_byte_size<HD>(n_kv_head);
}

// Same but returns a mutable pointer (for kernel self-increment of len).
template <int HD>
__device__ __forceinline__ uint8_t* get_slice_mut(uint64_t slices_ptr, int slice_idx, int n_kv_head) {
    return reinterpret_cast<uint8_t*>(slices_ptr)
        + (int64_t)slice_idx * token_slice_byte_size<HD>(n_kv_head);
}

__device__ __forceinline__ uint16_t slice_offset(const uint8_t* slice) {
    return *reinterpret_cast<const uint16_t*>(slice + 0);
}

__device__ __forceinline__ uint16_t slice_len(const uint8_t* slice) {
    return *reinterpret_cast<const uint16_t*>(slice + 2);
}

__device__ __forceinline__ uint32_t slice_rope(const uint8_t* slice) {
    return *reinterpret_cast<const uint32_t*>(slice + 4);
}

// Increment ws.len by 1. Called by the device-side post-decode commit kernel
// once the attention pass for the current token has completed.
__device__ __forceinline__ void slice_increment_len(uint8_t* slice) {
    // len is at byte offset 2 (uint16_t). The post-decode commit kernel uses
    // one thread per slot, so a plain non-atomic store is sufficient here.
    uint16_t cur = *reinterpret_cast<const uint16_t*>(slice + 2);
    *reinterpret_cast<uint16_t*>(slice + 2) = cur + 1;
}

// ============================================================================
// KvHead field accessors
// ============================================================================

// Get a pointer to KvHead[head_idx] within a slice.
template <int HD>
__device__ __forceinline__ const uint8_t* get_head(const uint8_t* slice, int head_idx) {
    return slice + 8 + (int64_t)head_idx * kv_head_byte_size<HD>();
}

// Mutable version.
template <int HD>
__device__ __forceinline__ uint8_t* get_head_mut(uint8_t* slice, int head_idx) {
    return slice + 8 + (int64_t)head_idx * kv_head_byte_size<HD>();
}

// k_pal map: HD/4 bytes at offset 0 within the head entry.
template <int HD>
__device__ __forceinline__ const uint8_t* kvhead_k_pal_map(const uint8_t* head) {
    return head;  // k_pal[] starts at offset 0
}

// v_pal map: HD/4 bytes at offset HD/4 within the head entry.
template <int HD>
__device__ __forceinline__ const uint8_t* kvhead_v_pal_map(const uint8_t* head) {
    return head + HD / 4;
}

// Compare two palette maps for equality. This is used to select the fast
// uniform-palette decode path when all slices in a sequence share the same
// routing, while still preserving correctness for reconciled slice-local maps.
template <int HD>
__device__ __forceinline__ bool pal_map_equal(const uint8_t* a, const uint8_t* b) {
    constexpr int N_U32 = (HD / 4) / 4;
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);
    #pragma unroll
    for (int i = 0; i < N_U32; ++i) {
        if (a32[i] != b32[i]) return false;
    }
    return true;
}

// k_ptr[p]: uint64_t at offset (HD/2 + p*8).
template <int HD>
__device__ __forceinline__ uint64_t kvhead_k_ptr(const uint8_t* head, int p) {
    return *reinterpret_cast<const uint64_t*>(head + HD / 2 + p * 8);
}

// v_ptr[p]: uint64_t at offset (HD/2 + 32 + p*8).
template <int HD>
__device__ __forceinline__ uint64_t kvhead_v_ptr(const uint8_t* head, int p) {
    return *reinterpret_cast<const uint64_t*>(head + HD / 2 + 32 + p * 8);
}

// k_fmt[p]: uint8_t at offset (HD/2 + 64 + p).
template <int HD>
__device__ __forceinline__ int kvhead_k_fmt(const uint8_t* head, int p) {
    return (int)*(head + HD / 2 + 64 + p);
}

// v_fmt[p]: uint8_t at offset (HD/2 + 68 + p).
template <int HD>
__device__ __forceinline__ int kvhead_v_fmt(const uint8_t* head, int p) {
    return (int)*(head + HD / 2 + 68 + p);
}

// k_scale[p]: float at offset (HD/2 + 72 + p*4).
template <int HD>
__device__ __forceinline__ float kvhead_k_scale(const uint8_t* head, int p) {
    return *reinterpret_cast<const float*>(head + HD / 2 + 72 + p * 4);
}

// v_scale[p]: float at offset (HD/2 + 88 + p*4).
template <int HD>
__device__ __forceinline__ float kvhead_v_scale(const uint8_t* head, int p) {
    return *reinterpret_cast<const float*>(head + HD / 2 + 88 + p * 4);
}
