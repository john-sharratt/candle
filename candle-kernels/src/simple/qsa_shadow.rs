//! FFI binding for the shadow scan (`simple/qsa_shadow.cu`).
//!
//! `run_qsa_shadow` scores each due sequence's kept queries against its
//! unloaded pooled rows, in windows seated behind each query, and counts a row
//! that is in its window's top `keep` and above the query's cut into the
//! round's counter — one launch over every due sequence, each named by one
//! [`ShadowSpan`] of a device table (`docs/hit_driven_projection.md` §6.3).
//!
//! The table is one upload: the spans, then every span's page descriptors
//! (`SHADOW_PAGE_WORDS` i64 each), then every span's row prefixes, which a
//! span names by word offsets from the table's start.

use std::ffi::c_void;

/// i64 words per page descriptor: `{keys, cstride, rstride, delta}`, the paged
/// scorer's. `delta` is unused.
pub const SHADOW_PAGE_WORDS: usize = 4;

/// The most index heads a kept query has (`QSA_SHADOW_MAX_HEADS`).
pub const SHADOW_MAX_HEADS: usize = 8;

/// The most `heads · head_dim` a kept query holds: the kernel's shared query
/// buffer.
pub const SHADOW_MAX_Q_WORDS: usize = SHADOW_MAX_HEADS * 256;

/// One due sequence's part of a shadow launch. Mirrors `ShadowSpan` in the
/// `.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShadowSpan {
    /// Device `[KEEP_SLOTS, q_words + CUT_WORDS]` u32: this layer's kept
    /// queries.
    pub keep: *const u32,
    /// Device `[n_rows]` u32: each unloaded row's counter.
    pub row_map: *const u32,
    /// Device counters: hits, queries, shadow queries, publish sequence.
    pub counters: *mut u32,
    /// Device factored RoPE table of the sequence's rung.
    pub tab: *const f32,
    /// Words from the table's start to this span's page descriptors. Even, so
    /// the i64 descriptors are aligned.
    pub pages_off: u32,
    /// Words from the table's start to this span's `[n_pages + 1]` row prefix.
    pub first_off: u32,
    pub n_rows: u32,
    pub n_pages: u32,
    pub n_pieces: u32,
    /// Kept slots 0..32 the scan reads.
    pub mask_lo: u32,
    /// Kept slots 32..64.
    pub mask_hi: u32,
    pub pad: u32,
}

/// Words one [`ShadowSpan`] is uploaded as.
pub const SHADOW_SPAN_WORDS: usize = 16;

impl ShadowSpan {
    /// The span as the kernel reads it: four pointers as low and high halves,
    /// then the eight counts and offsets — what a table of spans is uploaded
    /// as.
    pub fn words(&self) -> [u32; SHADOW_SPAN_WORDS] {
        let p = [
            self.keep as u64,
            self.row_map as u64,
            self.counters as u64,
            self.tab as u64,
        ];
        let mut w = [0u32; SHADOW_SPAN_WORDS];
        for (i, v) in p.iter().enumerate() {
            w[2 * i] = *v as u32;
            w[2 * i + 1] = (*v >> 32) as u32;
        }
        w[8..].copy_from_slice(&[
            self.pages_off,
            self.first_off,
            self.n_rows,
            self.n_pages,
            self.n_pieces,
            self.mask_lo,
            self.mask_hi,
            self.pad,
        ]);
        w
    }
}

extern "C" {
    /// `spans` is device memory holding `n_spans` [`ShadowSpan`] and the
    /// descriptors and prefixes they name; `max_windows` the most windows any
    /// kept query cuts its rows into.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_shadow(
        spans: *const c_void,
        n_spans: i32,
        max_windows: i32,
        heads: i32,
        head_dim: i32,
        pairs: i32,
        ratio: i32,
        top_k: i32,
        stream: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::{ShadowSpan, SHADOW_SPAN_WORDS};

    /// The layout the kernel reads, and the words a table is uploaded as are
    /// exactly the struct's bytes.
    #[test]
    fn a_span_is_the_kernels_layout() {
        assert_eq!(std::mem::size_of::<ShadowSpan>(), SHADOW_SPAN_WORDS * 4);
        assert_eq!(std::mem::align_of::<ShadowSpan>(), 8);
        let span = ShadowSpan {
            keep: 0x1000_0000_2000 as *const u32,
            row_map: 0x3000_0000_4000 as *const u32,
            counters: 0x5000_0000_6000 as *mut u32,
            tab: 0x7000_0000_8000 as *const f32,
            pages_off: 16,
            first_off: 40,
            n_rows: 1,
            n_pages: 2,
            n_pieces: 3,
            mask_lo: 4,
            mask_hi: 5,
            pad: 0,
        };
        // SAFETY: `ShadowSpan` is `repr(C)`, 64 bytes, with no padding.
        let bytes: [u8; SHADOW_SPAN_WORDS * 4] = unsafe { std::mem::transmute_copy(&span) };
        let from_words: Vec<u8> = span.words().iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes.to_vec(), from_words);
        assert_eq!(
            span.words(),
            [
                0x2000, 0x1000, 0x4000, 0x3000, 0x6000, 0x5000, 0x8000, 0x7000, 16, 40, 1, 2, 3, 4,
                5, 0
            ]
        );
    }
}
