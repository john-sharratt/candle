//! Copies of a slot's serialised state into a stager generation, so a launch
//! that is deferred reads a stable page table.
//!
//! The live slot can move to a wider class or be claimed by another sequence
//! before a deferred launch runs, so each per-token snapshot is copied into the
//! generation's arena. The records are copied once per epoch and chunk count and
//! reused, since they do not change within a chunk count, while the 16-byte
//! headers are copied on every call with their inline record pointers rebased
//! onto that copy. A debug fingerprint of the records catches a mutation that
//! would leave the cached copy stale.

use super::*;

/// A records-section copy living in a stager generation's arena.
pub(super) struct GenRecordsCache {
    /// Stager epoch the copy was made under; a mismatch means the arena reset.
    pub(super) epoch: u64,
    /// Chunk count the copy covers; a mismatch means a chunk was appended.
    pub(super) n_chunks: usize,
    /// Device pointer to the start of the copied records section.
    pub(super) dev_ptr: u64,
    /// FNV-1a fingerprint of the records bytes at copy time. Checked (debug only)
    /// on every cache hit to catch a record mutating within one (epoch, n_chunks).
    pub(super) records_checksum: u64,
}

/// Rebase inline `kvheads_ptr`s and patch the write-chunk length in a copied
/// `TokenSlice` header array. Pure byte arithmetic (no device, no unsafe) so it
/// is unit-testable against raw expected bytes.
///
/// `dst` is the freshly-copied header array to fix up; `src` is the authoritative
/// source header bytes to read the *original* pointers from (reading from `src`
/// rather than `dst` avoids a write-combined read-after-write hazard). For each
/// chunk, if its `kvheads_ptr` (u64 at header offset 8) falls in the source
/// records section `[records_base_old, records_end_old)` it is inline and is
/// rebased onto `records_ptr`; otherwise it is a resident/arena pointer and left
/// as copied. Finally the write chunk's `len` (u16 at header offset 2) is set to
/// `write_len`. Caller guarantees `write_idx < n` and `dst.len() == src.len() == n * 16`.
pub(super) fn rebase_and_patch_headers(
    dst: &mut [u8],
    src: &[u8],
    n: usize,
    records_base_old: u64,
    records_end_old: u64,
    records_ptr: u64,
    write_idx: usize,
    write_len: u16,
) {
    for i in 0..n {
        let off = i * SLICE_HEADER_BYTES + 8;
        let p = u64::from_le_bytes(src[off..off + 8].try_into().unwrap());
        if p >= records_base_old && p < records_end_old {
            let np = records_ptr + (p - records_base_old);
            dst[off..off + 8].copy_from_slice(&np.to_le_bytes());
        }
    }
    let loff = write_idx * SLICE_HEADER_BYTES + 2;
    dst[loff..loff + 2].copy_from_slice(&write_len.to_le_bytes());
}

/// FNV-1a checksum of a byte slice — a cheap fingerprint for the records-cache
/// staleness debug assertion (release builds never call it).
pub(super) fn records_checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Copy `bytes` into a fresh device-mapped bump buffer in `generation` and
/// return its device pointer. The copied bytes are read verbatim by the GPU.
pub(super) fn copy_into_generation(
    generation: &candle::quantized::pinned_staging::Generation,
    bytes: &[u8],
) -> candle::Result<u64> {
    let mut pinned = generation.alloc(bytes.len())?;
    if !pinned.is_bump() {
        candle::bail!(
            "copy_into_generation: expected a bump-allocated staging buffer for {} bytes",
            bytes.len()
        );
    }
    pinned.as_mut_slice().copy_from_slice(bytes);
    let gpu = generation.submit(pinned)?;
    Ok(gpu.dev_ptr())
}
