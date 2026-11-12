//! FFI binding for the QSA hit counter (`simple/qsa_hits.cu`).
//!
//! `run_qsa_count_hits` adds, for every block a selection row names, one hit
//! to the counter of the placed piece the block belongs to, and one query for
//! every row that selected — one launch over a whole wave's rows, each
//! sequence's rows, block→piece map and counters named by one [`HitSpan`] of a
//! device table (`docs/hit_driven_projection.md` §6.4). `run_qsa_publish_hits`
//! copies each counter set into its mapped summary block, where the host reads
//! it without waiting.

use std::ffi::c_void;

/// A block no placed piece owns.
pub const NO_PIECE: u32 = u32::MAX;

/// Words ahead of the counts in a summary block (`HIT_SUMMARY_HEADER`).
pub const SUMMARY_HEADER_WORDS: usize = 5;

/// Words a counter set holds past its hits: the queries counted, then the
/// publish sequence word.
pub const COUNTER_TAIL_WORDS: usize = 2;

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
    /// Hit counters the sequence holds.
    pub n_pieces: u32,
    /// Device `[map_len]` u32: block → piece, [`NO_PIECE`] for none.
    pub map: *const u32,
    /// Device `[n_pieces + COUNTER_TAIL_WORDS]` u32: hits, queries and the
    /// publish sequence, accumulated.
    pub counters: *mut u32,
    /// Device address of the mapped `[SUMMARY_HEADER_WORDS + n_pieces]` block.
    pub summary: *mut u32,
    /// The holder the summary belongs to.
    pub token: u32,
    /// The holder's declaration.
    pub generation: u32,
    /// 1 on the one span per counter set that publishes it.
    pub publish: u32,
    pub pad: u32,
}

/// Words one [`HitSpan`] is uploaded as.
pub const HIT_SPAN_WORDS: usize = 14;

impl HitSpan {
    /// The span as the kernel reads it, fourteen little-endian `u32` words:
    /// the four counts, each pointer's low and high halves, then the token,
    /// the generation, the publish flag and the padding — what a table of
    /// spans is uploaded as.
    pub fn words(&self) -> [u32; HIT_SPAN_WORDS] {
        let map = self.map as u64;
        let counters = self.counters as u64;
        let summary = self.summary as u64;
        [
            self.row_lo,
            self.row_hi,
            self.map_len,
            self.n_pieces,
            map as u32,
            (map >> 32) as u32,
            counters as u32,
            (counters >> 32) as u32,
            summary as u32,
            (summary >> 32) as u32,
            self.token,
            self.generation,
            self.publish,
            self.pad,
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

    /// Publish every span flagged `publish` into its summary block.
    pub fn run_qsa_publish_hits(spans: *const c_void, n_spans: i32, stream: *mut c_void);
}

#[cfg(test)]
mod tests {
    use super::{HitSpan, HIT_SPAN_WORDS};

    /// The layout the kernel reads: four u32, three pointers, four u32 — and
    /// the words a table is uploaded as are exactly the struct's bytes.
    #[test]
    fn a_span_is_the_kernels_layout() {
        assert_eq!(std::mem::size_of::<HitSpan>(), HIT_SPAN_WORDS * 4);
        assert_eq!(std::mem::align_of::<HitSpan>(), 8);
        let span = HitSpan {
            row_lo: 3,
            row_hi: 9,
            map_len: 40,
            n_pieces: 5,
            map: 0x1122_3344_5566_7788 as *const u32,
            counters: 0x99aa_bbcc_ddee_ff00 as *mut u32,
            summary: 0x0102_0304_0506_0708 as *mut u32,
            token: 11,
            generation: 12,
            publish: 1,
            pad: 0,
        };
        // SAFETY: `HitSpan` is `repr(C)`, 56 bytes, with no padding.
        let bytes: [u8; HIT_SPAN_WORDS * 4] = unsafe { std::mem::transmute_copy(&span) };
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
                0x99aa_bbcc,
                0x0506_0708,
                0x0102_0304,
                11,
                12,
                1,
                0
            ]
        );
    }
}
