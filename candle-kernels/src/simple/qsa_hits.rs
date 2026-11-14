//! FFI binding for the QSA hit counter (`simple/qsa_hits.cu`).
//!
//! `run_qsa_count_hits` adds, for every block a selection row names, one hit
//! to the counter of the declared round the block belongs to, and one query
//! for every row that selected — one launch over a whole wave's rows, each
//! sequence's rows, block→round map, counters and kept-query buffer named by
//! one [`HitSpan`] of a device table (`docs/hit_driven_projection.md` §6.4).
//! The same launch keeps the last [`KEEP_SLOTS`] rows of each span, query and
//! cut record, for the shadow scan. `run_qsa_publish_hits` copies each counter
//! set into its mapped summary block, where the host reads it without waiting.

use std::ffi::c_void;

/// A block no declared round owns.
pub const NO_PIECE: u32 = u32::MAX;

/// Words ahead of the counts in a summary block (`HIT_SUMMARY_HEADER`): begin,
/// token, generation, end, queries, shadow queries.
pub const SUMMARY_HEADER_WORDS: usize = 6;

/// Words a counter set holds past its hits: the queries counted, the shadow
/// queries counted, then the publish sequence word.
pub const COUNTER_TAIL_WORDS: usize = 3;

/// Kept queries per sequence per sampled layer (`QSA_KEEP_SLOTS`).
pub const KEEP_SLOTS: usize = 64;

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
    /// Device `[map_len]` u32: block → round, [`NO_PIECE`] for none.
    pub map: *const u32,
    /// Device `[n_pieces + COUNTER_TAIL_WORDS]` u32: hits, queries, shadow
    /// queries and the publish sequence, accumulated.
    pub counters: *mut u32,
    /// Device address of the mapped `[SUMMARY_HEADER_WORDS + n_pieces]` block.
    pub summary: *mut u32,
    /// Device `[layers, KEEP_SLOTS, q_words + CUT_WORDS]` u32: the kept
    /// queries, each its rotated query's f32 bits then its cut record.
    pub keep: *mut u32,
    /// The holder the summary belongs to.
    pub token: u32,
    /// The holder's declaration.
    pub generation: u32,
    /// 1 on the one span per counter set that publishes it.
    pub publish: u32,
    pub pad: u32,
}

/// Words one [`HitSpan`] is uploaded as.
pub const HIT_SPAN_WORDS: usize = 16;

impl HitSpan {
    /// The span as the kernel reads it, sixteen little-endian `u32` words:
    /// the four counts, each pointer's low and high halves, then the token,
    /// the generation, the publish flag and the padding — what a table of
    /// spans is uploaded as.
    pub fn words(&self) -> [u32; HIT_SPAN_WORDS] {
        let map = self.map as u64;
        let counters = self.counters as u64;
        let summary = self.summary as u64;
        let keep = self.keep as u64;
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
            keep as u32,
            (keep >> 32) as u32,
            self.token,
            self.generation,
            self.publish,
            self.pad,
        ]
    }
}

extern "C" {
    /// `spans` is device memory holding `n_spans` [`HitSpan`]; `max_rows` the
    /// most rows any span has. `q` is the wave's `[rows, q_words]` rotated
    /// queries, `cuts` its `[rows, CUT_WORDS]` cut records, and `layer` the
    /// sampled layer's index into each span's kept-query buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_count_hits(
        entries: *const u32,
        entry_stride: i32,
        cnt: *const u32,
        q: *const f32,
        q_words: i32,
        cuts: *const u32,
        layer: i32,
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

    /// The layout the kernel reads: four u32, four pointers, four u32 — and
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
            keep: 0x0a0b_0c0d_0e0f_1011 as *mut u32,
            token: 11,
            generation: 12,
            publish: 1,
            pad: 0,
        };
        // SAFETY: `HitSpan` is `repr(C)`, 64 bytes, with no padding.
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
                0x0e0f_1011,
                0x0a0b_0c0d,
                11,
                12,
                1,
                0
            ]
        );
    }
}
