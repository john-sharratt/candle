//! The per-sequence `SlotHeader` every paged attention kernel reads first
//! (`candle-kernels/src/paged-decode/slot_types.cuh`), as its writers build it.
//!
//! One record per sequence per launch, 32 bytes — one sector:
//!
//! ```text
//! [0..4)   u32 n_slices
//! [4..8)   u32 write_slice
//! [8..16)  u64 slices_ptr        device address of the slice table
//! [16..24) u64 position_map_ptr  prefill and glue only; 0 in decode
//! [24..28) u32 rope_rung         the sequence's RoPE rung
//! [28..32) u32 pad
//! ```
//!
//! **The rung is the sequence's own.** Every attention kernel maps a CTA to one
//! sequence and reads its rotation from that sequence's header, so a launch
//! serving sequences on different rungs shares nothing rung-dependent between
//! them (`docs/progressive_yarn.md` §6). Every writer serialises through
//! [`SlotHeaderHost::write`], so the layout lives in one place on this side.

/// Bytes of one `SlotHeader`; mirrors `sizeof(SlotHeader)` in `slot_types.cuh`.
pub const SLOT_HEADER_BYTES: usize = 32;

/// One sequence's header, as the host builds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotHeaderHost {
    pub n_slices: u32,
    pub write_slice: u32,
    pub slices_ptr: u64,
    pub position_map_ptr: u64,
    pub rope_rung: u32,
}

impl SlotHeaderHost {
    /// Append the header's 32 bytes, little-endian, in the kernel's layout.
    pub fn write(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.n_slices.to_le_bytes());
        buf.extend_from_slice(&self.write_slice.to_le_bytes());
        buf.extend_from_slice(&self.slices_ptr.to_le_bytes());
        buf.extend_from_slice(&self.position_map_ptr.to_le_bytes());
        buf.extend_from_slice(&self.rope_rung.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field lands at the offset `slot_types.cuh` asserts, and the
    /// record is exactly one sector.
    #[test]
    fn the_layout_is_the_kernels() {
        let h = SlotHeaderHost {
            n_slices: 0x0403_0201,
            write_slice: 0x0807_0605,
            slices_ptr: 0x100f_0e0d_0c0b_0a09,
            position_map_ptr: 0x1817_1615_1413_1211,
            rope_rung: 0x1c1b_1a19,
        };
        let mut buf = Vec::new();
        h.write(&mut buf);
        assert_eq!(buf.len(), SLOT_HEADER_BYTES);
        let want: Vec<u8> = (1u8..=28).chain([0, 0, 0, 0]).collect();
        assert_eq!(buf, want);
    }
}
