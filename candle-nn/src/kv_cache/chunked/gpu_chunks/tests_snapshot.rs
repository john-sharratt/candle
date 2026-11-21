//! Unit tests for the header arithmetic that rebases inline record pointers and
//! patches the write length.
//!
//! The rebase is pure byte arithmetic over raw 16-byte headers, so these tests
//! assert the bytes that come out: inline pointers move onto the relocated
//! records, resident pointers are left alone, only the write chunk's length
//! changes, and the bound on the last record is exact. A mutated record must
//! also change the checksum.

use super::*;

// Integration-style unit tests live in tests/gpu_chunks_tests.rs. The pure
// header-rebase arithmetic is tested here (no device needed).
#[cfg(test)]
pub(super) mod snapshot_tests {
    use super::{rebase_and_patch_headers, records_checksum, SLICE_HEADER_BYTES};

    /// Serialise one 16-byte `TokenSlice` header: offset u16 | len u16 | rope u32
    /// | kvheads_ptr u64.
    fn header(offset: u16, len: u16, rope: u32, kvheads_ptr: u64) -> [u8; 16] {
        let mut h = [0u8; 16];
        h[0..2].copy_from_slice(&offset.to_le_bytes());
        h[2..4].copy_from_slice(&len.to_le_bytes());
        h[4..8].copy_from_slice(&rope.to_le_bytes());
        h[8..16].copy_from_slice(&kvheads_ptr.to_le_bytes());
        h
    }

    fn read_ptr(bytes: &[u8], chunk: usize) -> u64 {
        let off = chunk * SLICE_HEADER_BYTES + 8;
        u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap())
    }
    fn read_len(bytes: &[u8], chunk: usize) -> u16 {
        let off = chunk * SLICE_HEADER_BYTES + 2;
        u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap())
    }

    #[test]
    fn rebase_inline_leaves_resident_and_patches_write_len() {
        // 3 chunks, 8-byte records; source buffer based at 0x10000, headers
        // occupy [0, 48), records [48, 72). Chunk 0/1 inline (kvheads_ptr into
        // the records section), chunk 2 resident (arena pointer, out of range).
        const REC_BYTES: u64 = 8;
        let d_old: u64 = 0x1_0000;
        let n = 3usize;
        let headers_len = n * SLICE_HEADER_BYTES; // 48
        let records_base_old = d_old + headers_len as u64; // 0x10030
        let records_end_old = d_old + headers_len as u64 + n as u64 * REC_BYTES; // 0x10048
        let records_ptr: u64 = 0x2_0000; // relocated records copy base

        let mut src = Vec::new();
        src.extend_from_slice(&header(0, 32, 0, records_base_old)); // chunk 0 inline
        src.extend_from_slice(&header(0, 5, 32, records_base_old + REC_BYTES)); // chunk 1 inline
        src.extend_from_slice(&header(7, 0, 64, 0x9999_9999_9999)); // chunk 2 resident
        let mut dst = src.clone();

        rebase_and_patch_headers(
            &mut dst,
            &src,
            n,
            records_base_old,
            records_end_old,
            records_ptr,
            1,  // write chunk
            18, // new write length
        );

        // Inline pointers rebased onto the relocated records copy.
        assert_eq!(read_ptr(&dst, 0), records_ptr);
        assert_eq!(read_ptr(&dst, 1), records_ptr + REC_BYTES);
        // Resident pointer untouched.
        assert_eq!(read_ptr(&dst, 2), 0x9999_9999_9999);
        // Only the write chunk's length changed.
        assert_eq!(read_len(&dst, 0), 32);
        assert_eq!(read_len(&dst, 1), 18);
        assert_eq!(read_len(&dst, 2), 0);
        // Non-pointer, non-write-len bytes of the write header are preserved
        // (offset stays 0, rope stays 32).
        assert_eq!(&dst[16..18], &0u16.to_le_bytes()); // chunk 1 offset
        assert_eq!(&dst[20..24], &32u32.to_le_bytes()); // chunk 1 rope
    }

    #[test]
    fn edge_case_last_chunk_inline_ptr_is_in_range() {
        // The last inline record starts at records_end_old - REC_BYTES, which
        // must still satisfy the `< records_end_old` bound (no off-by-one).
        const REC_BYTES: u64 = 8;
        let d_old: u64 = 0;
        let n = 1usize;
        let records_base_old = d_old + SLICE_HEADER_BYTES as u64; // 16
        let records_end_old = records_base_old + REC_BYTES; // 24
        let records_ptr: u64 = 0x5000;
        let src = header(0, 1, 0, records_base_old).to_vec();
        let mut dst = src.clone();
        rebase_and_patch_headers(
            &mut dst,
            &src,
            n,
            records_base_old,
            records_end_old,
            records_ptr,
            0,
            1,
        );
        assert_eq!(read_ptr(&dst, 0), records_ptr);
    }

    #[test]
    fn checksum_detects_record_mutation() {
        let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut b = a;
        b[3] = 0xff;
        assert_eq!(records_checksum(&a), records_checksum(&a));
        assert_ne!(records_checksum(&a), records_checksum(&b));
    }
}
