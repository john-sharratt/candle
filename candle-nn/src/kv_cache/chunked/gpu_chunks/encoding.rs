//! The serialised byte layout of a chunk's slot state, which the attention
//! kernel reads directly.
//!
//! Each chunk is a 16-byte slice header pointing at an out-of-line `KvHead`
//! record, and all headers sit in one contiguous array so the kernel can index
//! them by stride. Sizes follow from the head dimension and the record layout,
//! and the identity palette map packs four dimensions to a byte. The decode
//! buffer and the resident meta-pool both serialise through the same record
//! writer, so they produce byte-identical records.

use super::*;

// Serialization helpers

/// Returns the serialised byte-size of one `KvHead` entry.
///
/// Layout: `k_pal[head_dim/4] + v_pal[head_dim/4] + k_ptr[4]×8 + v_ptr[4]×8 + k_fmt[4] + v_fmt[4] + k_scale[4]×4 + v_scale[4]×4`
///
/// The pal_map packs 4 dims/byte (2 bits per dim), so each side is `head_dim/4`
/// bytes. The `/4` is the bit-packing density, *not* N_PALETTE — they
/// coincidentally both equal 4 today but are independent constants.
pub(crate) fn kv_head_serialized_size(head_dim: usize, n_palette: usize) -> usize {
    (head_dim / 4) * 2 // k_pal + v_pal, each head_dim/4 bytes (2-bit density)
        + n_palette * 26 // k_ptr+v_ptr (8 each) + k_fmt+v_fmt (1 each) + k_scale+v_scale (4 each)
}

/// Fixed byte-size of one `TokenSlice` header: offset(2) + len(2) + rope(4) +
/// kvheads_ptr(8). The KvHead record lives out-of-line (see [`record_bytes`]).
pub(crate) const SLICE_HEADER_BYTES: usize = 16;

/// Byte-size of one chunk's out-of-line `KvHead[n_kv_head]` record.
pub(crate) fn record_bytes(n_kv_head: usize, head_dim: usize, n_palette: usize) -> usize {
    n_kv_head * kv_head_serialized_size(head_dim, n_palette)
}

/// Per-chunk footprint in the `GpuChunks` buffer: the 16-byte slice header plus
/// its out-of-line record. The buffer is laid out in two sections —
/// `[ slice_header × capacity | record × capacity ]` — so slice headers stay a
/// contiguous 16-byte-stride array (what the kernel's `get_slice` indexes) while
/// each header's `kvheads_ptr` points into the records section.
pub(crate) fn token_slice_serialized_size(
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
) -> usize {
    SLICE_HEADER_BYTES + record_bytes(n_kv_head, head_dim, n_palette)
}

/// Write a 16-byte slice header (`offset`, `len`, `rope`, `kvheads_ptr`).
pub(crate) fn write_slice_header(
    dst: &mut [u8],
    offset: u16,
    len: u16,
    rope: u32,
    kvheads_ptr: u64,
) {
    dst[0..2].copy_from_slice(&offset.to_le_bytes());
    dst[2..4].copy_from_slice(&len.to_le_bytes());
    dst[4..8].copy_from_slice(&rope.to_le_bytes());
    dst[8..16].copy_from_slice(&kvheads_ptr.to_le_bytes());
}

/// Serialize one chunk's `KvHead[n_kv_head]` record into `dst` (length
/// [`record_bytes`]). Delegates to the shared record serializer so the decode
/// buffer and the resident meta-pool produce byte-identical records.
/// Per-head band count for this chunk's KvHead record, derived from its GID
/// count (`n_kv_head * n_palette * 2`): 8 for a single-latent chunk, 4 for GQA.
/// Falls back to [`N_PALETTE`] for an empty/placeholder chunk.
pub(super) fn chunk_n_palette(chunk: &ChunkWindow, n_kv_head: usize) -> usize {
    let g = chunk.gids.len();
    if n_kv_head == 0 || g == 0 {
        crate::kv_cache::arena_table::N_PALETTE
    } else {
        (g / (n_kv_head * 2)).max(1)
    }
}

pub(super) fn write_record_for_chunk(
    dst: &mut [u8],
    chunk: &ChunkWindow,
    n_kv_head: usize,
    head_dim: usize,
    n_palette: usize,
    arena_info: &[ResolvedArenaInfo],
) {
    super::super::meta_pool::serialize_kv_heads(
        dst,
        &super::super::meta_pool::ChunkRecordSrc {
            gids: &chunk.gids,
            k_pal: chunk.k_pal.as_slice(),
            v_pal: chunk.v_pal.as_slice(),
            k_scale: chunk.k_scale.as_slice(),
            v_scale: chunk.v_scale.as_slice(),
            k_fmt: chunk.k_fmt.as_slice(),
            v_fmt: chunk.v_fmt.as_slice(),
        },
        n_kv_head,
        head_dim,
        n_palette,
        arena_info,
    );
}

/// Write the identity 2-bit palette map for the given head dimension into `dst`.
///
/// Assigns dim `d` to palette `d / (head_dim / N_PALETTE)`, packed 4 dims
/// per byte in little-endian order: `(d3<<6)|(d2<<4)|(d1<<2)|d0`.
/// For N_PALETTE=4 this produces the (0,0,1,1,2,2,3,3,...) pattern.
/// `dst` must be exactly `(head_dim / 4).max(1)` bytes long.
#[cfg(test)]
pub(crate) fn write_identity_pal_map(head_dim: usize, dst: &mut [u8]) {
    let sub_hd = (head_dim / N_PALETTE).max(1);
    dst.fill(0);
    for d in 0..head_dim {
        let pal_idx = (d / sub_hd).min(N_PALETTE - 1) as u8;
        let byte_idx = d / 4;
        let bit_shift = (d % 4) * 2;
        dst[byte_idx] |= pal_idx << bit_shift;
    }
}
