#pragma once
// ============================================================================
// Persistent Slot Buffer Type Accessors
//
// Provides byte-layout accessors for the variable-size structures serialized
// by SlotPool on the Rust side and consumed by the v2 decode kernel.
//
// Byte layouts must match the Rust serialization in slot_state.rs exactly.
//
// SlotState header (16 bytes, fixed):
//   [0..4)   uint32_t n_slices
//   [4..8)   uint32_t write_slice
//   [8..16)  uint64_t slices_ptr       (device pointer into slices tensor)
//
// There is no position lookup table: `resolve_pos` below computes a position's
// (slice_idx, in_blk) from the slice array, which already carries everything
// such a table encoded (rope, len, offset) plus write_slice from this header.
//
// TokenSlice (16 bytes, fixed stride):
//   [0..2)   uint16_t offset
//   [2..4)   uint16_t len        ← committed on-device after each decode step
//   [4..8)   uint32_t rope
//   [8..16)  uint64_t kvheads_ptr ← device pointer to this chunk's
//                                   KvHead[n_kv_head] record (a resident meta
//                                   slab for sealed chunks, or a separate
//                                   records region the host uploaded alongside
//                                   the slices for transient/float chunks).
//
// KvHead (HD/2 + 104 bytes for HEAD_DIM HD) — out-of-line, pointed to by
// kvheads_ptr; byte layout unchanged so kvhead_* accessors / palette4_convert
// keep reading it identically:
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

#ifndef CHUNK_SIZE
#define CHUNK_SIZE 32
#endif

// ============================================================================
// SlotState header layout
// ============================================================================

struct SlotHeader {
    uint32_t n_slices;
    uint32_t write_slice;
    uint64_t slices_ptr;  // device pointer into per-slot slice data
};
static_assert(sizeof(SlotHeader) == 16, "SlotHeader must be 16 bytes");

// Read the SlotHeader for a given slot index from the headers tensor.
__device__ __forceinline__ const SlotHeader& get_slot_header(const uint8_t* headers, int slot_idx) {
    return *reinterpret_cast<const SlotHeader*>(headers + (int64_t)slot_idx * 16);
}


// ============================================================================
// Byte-size helpers (compile-time for HEAD_DIM, runtime for n_kv_head)
// ============================================================================

// Byte size of one KvHead entry for a given HEAD_DIM and palette count NP.
// Layout: k_pal[HD/4] + v_pal[HD/4] + k_ptr[NP]*8 + v_ptr[NP]*8 + k_fmt[NP] +
// v_fmt[NP] + k_scale[NP]*4 + v_scale[NP]*4 = HD/2 + NP*26. NP defaults to 4
// (GQA / palette4); the single-latent path instantiates NP = LATENT_N_BANDS.
template <int HD, int NP = 4>
__device__ __host__ constexpr int kv_head_byte_size() {
    return HD / 2 + NP * 26;
}

// Byte size of one TokenSlice — fixed at 16 (offset/len/rope + kvheads_ptr).
// The KvHead[n_kv_head] record is out-of-line behind kvheads_ptr. The n_kv_head
// parameter is kept so the (many) get_slice call sites need no change.
template <int HD>
__device__ __forceinline__ int token_slice_byte_size(int /*n_kv_head*/) {
    return 16;
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

// Resolve a cum_token position to (slice_idx, in_blk) from the slot's slices.
//
// Replaces `chunk_div(k_pos)` / `chunk_mod(k_pos)`, which worked only while
// every chunk held exactly CHUNK_SIZE tokens; partial chunks, injected
// substrate windows and glue gaps made chunks variable-length and broke the
// division.  The host answered that by precomputing a `u32[total_tokens]`
// position_map — one entry per token, rebuilt and pushed over PCIe every
// forward — and this computes the same answer from state already on the device.
//
// **Two regions, because the slot has two.**
//
// *Committed* tokens live in slices that record what they hold: `rope` is a
// slice's first cum_token position and `len` how many follow, so the owning
// slice is the one with `rope <= k_pos < rope + len` and the offset inside its
// chunk is `offset + (k_pos - rope)`.  Slices are ordered by `rope`, so this is
// a binary search.  Empty slices (`len == 0`) claim no position and the same
// comparison steps over them.
//
// *Pending writes* sit past the committed total, in capacity no slice counts
// yet, so they cannot be found by searching `len`.  They are laid out by
// walking forward from `write_slice` — filling the writer's chunk from
// `offset + len` to CHUNK_SIZE, then each following chunk from its own
// `offset`.  Every input it needs is here: `write_slice` is in the header and
// CHUNK_SIZE is a shared constant.  The host mirrors this walk in
// `resolve_pos_reference`, and `SlotStateHost::assert_write_region_capacity`
// refuses a forward whose write region the walk would run off the end of.
//
// **Why compute rather than look up.**  A table has to be kept consistent with
// what it describes, and this one had to be identical across all 48 layers to
// be shared — an invariance nothing structurally guaranteed.  A windowed creep
// prefill leaves layer 0 holding a trailing empty writer chunk its peers have
// not pushed, which shifts every later slice index; the launch guard that
// caught the mismatch cost run CM 102 of 354 directories.  Computed here there
// is no table to be stale, no invariance to establish, and no guard to trip.
// It also reads `len` as committed on-device after each decode step, where the
// map held what the host believed before the forward began.
//
// The cost is a handful of L1 reads over an array of tens of 16-byte slices,
// at call sites that resolve once per warp-column or once per tile rather than
// once per lane — against a per-forward host build and upload the map needed.
__device__ __forceinline__ void resolve_pos(
    const SlotHeader& slot_hdr,
    int k_pos,
    int& slice_idx,
    int& in_blk
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(slot_hdr.slices_ptr);
    const int n = (int)slot_hdr.n_slices;
    if (n <= 0) { slice_idx = 0; in_blk = 0; return; }

    int lo = 0;
    int hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint8_t* s = base + (int64_t)mid * 16;
        const int start = (int)slice_rope(s);
        const int len = (int)slice_len(s);
        if (k_pos < start) {
            hi = mid - 1;
        } else if (k_pos >= start + len) {
            lo = mid + 1;
        } else {
            slice_idx = mid;
            in_blk = (int)slice_offset(s) + (k_pos - start);
            return;
        }
    }

    // Past every committed token: a pending write. Walk from the writer.
    const uint8_t* last = base + (int64_t)(n - 1) * 16;
    const int committed = (int)slice_rope(last) + (int)slice_len(last);
    int cur = (int)slot_hdr.write_slice;
    if (cur >= n) { slice_idx = 0; in_blk = 0; return; }
    const uint8_t* ws = base + (int64_t)cur * 16;
    int cur_in_blk = (int)slice_offset(ws) + (int)slice_len(ws);
    int j = k_pos - committed;
    while (true) {
        if (cur_in_blk < CHUNK_SIZE) {
            const int cap = CHUNK_SIZE - cur_in_blk;
            if (j < cap) {
                slice_idx = cur;
                in_blk = cur_in_blk + j;
                return;
            }
            j -= cap;
        }
        cur += 1;
        if (cur >= n) { slice_idx = 0; in_blk = 0; return; }
        cur_in_blk = (int)slice_offset(base + (int64_t)cur * 16);
    }
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

// Get a pointer to KvHead[head_idx] for a slice. The slice stores a device
// pointer to its KvHead[n_kv_head] record at byte offset 8; dereference and
// index by head.
template <int HD, int NP = 4>
__device__ __forceinline__ const uint8_t* get_head(const uint8_t* slice, int head_idx) {
    uint64_t kvheads_ptr = *reinterpret_cast<const uint64_t*>(slice + 8);
    return reinterpret_cast<const uint8_t*>(kvheads_ptr)
        + (int64_t)head_idx * kv_head_byte_size<HD, NP>();
}

// Mutable version.
template <int HD, int NP = 4>
__device__ __forceinline__ uint8_t* get_head_mut(uint8_t* slice, int head_idx) {
    uint64_t kvheads_ptr = *reinterpret_cast<const uint64_t*>(slice + 8);
    return reinterpret_cast<uint8_t*>(kvheads_ptr)
        + (int64_t)head_idx * kv_head_byte_size<HD, NP>();
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

// The palette block starts at HD/2 (after k_pal + v_pal). Slot offsets within
// it are parameterized on NP so a single-latent record (NP = LATENT_N_BANDS)
// and a GQA record (NP = 4, the default) share one accessor family:
//   k_ptr  @ HD/2 + p*8          v_ptr  @ HD/2 + NP*8 + p*8
//   k_fmt  @ HD/2 + NP*16 + p    v_fmt  @ HD/2 + NP*17 + p
//   k_scale@ HD/2 + NP*18 + p*4  v_scale@ HD/2 + NP*22 + p*4

// k_ptr[p]: uint64_t at offset (HD/2 + p*8).
template <int HD, int NP = 4>
__device__ __forceinline__ uint64_t kvhead_k_ptr(const uint8_t* head, int p) {
    return *reinterpret_cast<const uint64_t*>(head + HD / 2 + p * 8);
}

// v_ptr[p]: uint64_t at offset (HD/2 + NP*8 + p*8).
template <int HD, int NP = 4>
__device__ __forceinline__ uint64_t kvhead_v_ptr(const uint8_t* head, int p) {
    return *reinterpret_cast<const uint64_t*>(head + HD / 2 + NP * 8 + p * 8);
}

// k_fmt[p]: uint8_t at offset (HD/2 + NP*16 + p).
template <int HD, int NP = 4>
__device__ __forceinline__ int kvhead_k_fmt(const uint8_t* head, int p) {
    return (int)*(head + HD / 2 + NP * 16 + p);
}

// v_fmt[p]: uint8_t at offset (HD/2 + NP*17 + p).
template <int HD, int NP = 4>
__device__ __forceinline__ int kvhead_v_fmt(const uint8_t* head, int p) {
    return (int)*(head + HD / 2 + NP * 17 + p);
}

// k_scale[p]: float at offset (HD/2 + NP*18 + p*4).
template <int HD, int NP = 4>
__device__ __forceinline__ float kvhead_k_scale(const uint8_t* head, int p) {
    return *reinterpret_cast<const float*>(head + HD / 2 + NP * 18 + p * 4);
}

// v_scale[p]: float at offset (HD/2 + NP*22 + p*4).
template <int HD, int NP = 4>
__device__ __forceinline__ float kvhead_v_scale(const uint8_t* head, int p) {
    return *reinterpret_cast<const float*>(head + HD / 2 + NP * 22 + p * 4);
}
