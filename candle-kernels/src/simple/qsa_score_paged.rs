//! FFI binding for the ragged, paged QSA index scorer.
//!
//! Scores a wave's indexer queries against an index that is **not one buffer**:
//! the rows live in per-turn pages plus the live tail, each separately
//! allocated, and the last row of a page covers fewer than `ratio` tokens
//! because a turn boundary does not land on a block boundary. Keys are stored
//! un-rotated, and the kernel rotates each one as it loads it. See
//! `simple/qsa_score_paged.cu`.
//!
//! ```text
//!   out[r, g] = Σ_h relu( q[r, h, :] · rope(key_g, delta_p + g·ratio) )   for g <  cnt[r]
//!             = -1e30                                                    for g >= cnt[r]
//! ```
//!
//!   q           device f32[rows * H * D], contiguous, rotated, `[rows*H, D]`
//!   pages       device i64[P * PAGE_WORDS] — `{keys, cstride, rstride, delta}`,
//!               strides in `float4`s
//!   page_first  device u32[P+1] — exclusive prefix sum of page row counts
//!   cnt         device u32[rows] — valid candidate prefix per query row
//!   tab         the factored RoPE table, f32[(2048 + 1024) · pairs · 2]
//!   steps       the step table for a stride of `ratio`, f32[pairs · 32 · 2]
//!   out         device f32, row `r` at `(row_base + r) * out_s`
//!
//! **`cnt` carries the ragged widths.** Rows stay ordered by token position, so
//! "wholly below this query" is still a prefix however wide each row is; the
//! caller folds the per-page widths into `cnt` on the host.

use std::ffi::c_void;

/// i64 words per page descriptor. Mirrors `QSA_PAGED_PAGE_WORDS`.
pub const PAGE_WORDS: usize = 4;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_score_paged(
        q: *const f32,
        pages: *const i64,
        page_first: *const u32,
        cnt: *const u32,
        tab: *const f32,
        steps: *const f32,
        out: *mut f32,
        rows: i32,
        h: i32,
        d: i32,
        n_cand: i32,
        p: i32,
        pairs: i32,
        ratio: i32,
        out_s: i64,
        row_base: i64,
        stream: *mut c_void,
    );
}
