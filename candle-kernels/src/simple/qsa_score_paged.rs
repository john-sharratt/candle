//! FFI binding for the ragged, paged QSA index scorer.
//!
//! Scores a wave's indexer queries against an index that is **not one buffer**:
//! the rows live in per-turn pages, each separately allocated, and the last row
//! of a page covers fewer than `ratio` tokens because a turn boundary does not
//! land on a block boundary. See `simple/qsa_score_paged.cu`.
//!
//! ```text
//!   out[r, g] = Σ_h relu( q[r, h, :] · key_g[:] )   for g <  cnt[r]
//!             = -1e30                               for g >= cnt[r]
//! ```
//!
//!   q           device f32[rows * H * D], contiguous, row-major `[rows*H, D]`
//!   page_keys   device u64[P] — each page's key buffer address
//!   page_first  device u32[P+1] — exclusive prefix sum of page row counts
//!   cnt         device u32[rows] — valid candidate prefix per query row
//!   out         device f32, row `r` at `(row_base + r) * out_s`
//!
//! **`cnt` carries the ragged widths.** Rows stay ordered by token position, so
//! "wholly below this query" is still a prefix however wide each row is; the
//! caller folds the per-page widths into `cnt` on the host, where the page table
//! already lives. The kernel therefore never sees a width or a position.

use std::ffi::c_void;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_score_paged(
        q: *const f32,
        page_keys: *const u64,
        page_first: *const u32,
        cnt: *const u32,
        out: *mut f32,
        rows: i32,
        h: i32,
        d: i32,
        n_cand: i32,
        p: i32,
        out_s: i64,
        row_base: i64,
        stream: *mut c_void,
    );
}
