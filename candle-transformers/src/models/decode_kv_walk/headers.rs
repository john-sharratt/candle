//! The two fixed-size records the paged decode kernel walks before it reaches
//! any KV: the per-row `SlotHeader` and the per-chunk `TokenSlice`.
//!
//! Byte layouts are those of `paged-decode/slot_types.cuh`, read back exactly as
//! the kernel reads them.

use candle::Result;

/// Bytes per `SlotHeader` and per `TokenSlice`.
pub(crate) const RECORD_BYTES: usize = 16;

/// One decode row's `SlotHeader`: `{n_slices: u32, write_slice: u32, slices_ptr: u64}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotHeader {
    pub n_slices: u32,
    pub write_slice: u32,
    pub slices_ptr: u64,
}

/// One chunk's `TokenSlice`: `{offset: u16, len: u16, rope: u32, kvheads_ptr: u64}`.
///
/// `offset..offset + len` is the token window inside the chunk that the kernel
/// attends; `kvheads_ptr` is the device address of the chunk's
/// `KvHead[n_kv_head]` record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TokenSlice {
    pub offset: u16,
    pub len: u16,
    pub rope: u32,
    pub kvheads_ptr: u64,
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"))
}
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"))
}

/// Parse a 16-byte `SlotHeader`.
pub(crate) fn parse_slot_header(b: &[u8]) -> Result<SlotHeader> {
    if b.len() != RECORD_BYTES {
        candle::bail!("slot header: {} bytes, want {RECORD_BYTES}", b.len());
    }
    Ok(SlotHeader {
        n_slices: u32_at(b, 0),
        write_slice: u32_at(b, 4),
        slices_ptr: u64_at(b, 8),
    })
}

/// Parse a packed array of `TokenSlice`s.
pub(crate) fn parse_slices(b: &[u8]) -> Result<Vec<TokenSlice>> {
    if !b.len().is_multiple_of(RECORD_BYTES) {
        candle::bail!(
            "token slices: {} bytes is not a whole number of 16-byte slices",
            b.len()
        );
    }
    Ok(b.chunks_exact(RECORD_BYTES)
        .map(|s| TokenSlice {
            offset: u16_at(s, 0),
            len: u16_at(s, 2),
            rope: u32_at(s, 4),
            kvheads_ptr: u64_at(s, 8),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slot_header_reads_its_three_fields_little_endian() {
        let mut b = Vec::new();
        b.extend_from_slice(&7u32.to_le_bytes());
        b.extend_from_slice(&6u32.to_le_bytes());
        b.extend_from_slice(&0xcc3e_8000_0000u64.to_le_bytes());
        assert_eq!(
            parse_slot_header(&b).unwrap(),
            SlotHeader {
                n_slices: 7,
                write_slice: 6,
                slices_ptr: 0xcc3e_8000_0000
            }
        );
    }

    #[test]
    fn a_slot_header_of_the_wrong_size_is_refused() {
        assert!(parse_slot_header(&[0u8; 15]).is_err());
    }

    #[test]
    fn slices_read_back_in_order() {
        let mut b = Vec::new();
        for (off, len, rope, ptr) in [(0u16, 32u16, 0u32, 0x1000u64), (4, 9, 32, 0x2000)] {
            b.extend_from_slice(&off.to_le_bytes());
            b.extend_from_slice(&len.to_le_bytes());
            b.extend_from_slice(&rope.to_le_bytes());
            b.extend_from_slice(&ptr.to_le_bytes());
        }
        assert_eq!(
            parse_slices(&b).unwrap(),
            vec![
                TokenSlice {
                    offset: 0,
                    len: 32,
                    rope: 0,
                    kvheads_ptr: 0x1000
                },
                TokenSlice {
                    offset: 4,
                    len: 9,
                    rope: 32,
                    kvheads_ptr: 0x2000
                },
            ]
        );
    }

    #[test]
    fn a_partial_slice_is_refused() {
        assert!(parse_slices(&[0u8; 17]).is_err());
    }
}
