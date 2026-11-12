//! FFI binding for the QSA hit counter (`simple/qsa_hits.cu`).
//!
//! `run_qsa_count_hits` adds, for every block a selection row names, one hit
//! to the counter of the placed piece the block belongs to — one launch over a
//! whole wave's rows, each sequence's rows, block→piece map and counters named
//! by one [`HitSpan`] of a device table (`docs/hit_driven_projection.md` §6.4).

use std::ffi::c_void;

/// A block no placed piece owns.
pub const NO_PIECE: u32 = u32::MAX;

/// One sequence's part of a hit-count launch. Mirrors `HitSpan` in the `.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HitSpan {
    /// First row of the sequence in the launch.
    pub row_lo: u32,
    /// One past its last row.
    pub row_hi: u32,
    /// Blocks the map covers.
    pub map_len: u32,
    /// Counters the sequence holds.
    pub n_pieces: u32,
    /// Device `[map_len]` u32: block → piece, [`NO_PIECE`] for none.
    pub map: *const u32,
    /// Device `[n_pieces]` u32: hits, accumulated.
    pub counters: *mut u32,
}

impl HitSpan {
    /// The span as the kernel reads it, eight little-endian `u32` words:
    /// the four counts, then each pointer's low and high halves — what a
    /// table of spans is uploaded as.
    pub fn words(&self) -> [u32; 8] {
        let map = self.map as u64;
        let counters = self.counters as u64;
        [
            self.row_lo,
            self.row_hi,
            self.map_len,
            self.n_pieces,
            map as u32,
            (map >> 32) as u32,
            counters as u32,
            (counters >> 32) as u32,
        ]
    }
}

extern "C" {
    /// `spans` is device memory holding `n_spans` [`HitSpan`]; `max_rows` the
    /// most rows any span has.
    pub fn run_qsa_count_hits(
        entries: *const u32,
        entry_stride: i32,
        cnt: *const u32,
        spans: *const c_void,
        n_spans: i32,
        max_rows: i32,
        stream: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::HitSpan;

    /// The layout the kernel reads: four u32 then two pointers — and the
    /// words a table is uploaded as are exactly the struct's bytes.
    #[test]
    fn a_span_is_the_kernels_layout() {
        assert_eq!(std::mem::size_of::<HitSpan>(), 16 + 2 * 8);
        assert_eq!(std::mem::align_of::<HitSpan>(), 8);
        let span = HitSpan {
            row_lo: 3,
            row_hi: 9,
            map_len: 40,
            n_pieces: 5,
            map: 0x1122_3344_5566_7788 as *const u32,
            counters: 0x99aa_bbcc_ddee_ff00 as *mut u32,
        };
        // SAFETY: `HitSpan` is `repr(C)`, 32 bytes, with no padding.
        let bytes: [u8; 32] = unsafe { std::mem::transmute_copy(&span) };
        let from_words: Vec<u8> = span.words().iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes.to_vec(), from_words);
        assert_eq!(
            span.words(),
            [
                3,
                9,
                40,
                5,
                0x5566_7788,
                0x1122_3344,
                0xddee_ff00,
                0x99aa_bbcc
            ]
        );
    }
}
