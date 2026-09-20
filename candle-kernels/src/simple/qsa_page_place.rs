//! FFI binding for QSA index page placement.
//!
//! Writes a page's un-rotated block keys in the scorer's channel-blocked
//! layout. There is no rotation: the scorer rotates each key as it loads it.
//! See `simple/qsa_page_place.cu`.
//!
//! ```text
//!   src   device f32[rows, d]        the page's keys, as a record holds them
//!   dst   device f32[d/4, rows, 4]   the page's placement
//! ```
//!
//! Batched over a descriptor table: one job per page.

use std::ffi::c_void;

/// i64 words per job: `{src, dst, rows}`.
pub const PLACE_JOB_WORDS: usize = 3;

extern "C" {
    /// `tile_r` selects the row tile; `0` takes the tuned default. The explicit
    /// values exist for the sweep in `tests/qsa_page_place_bench.rs`.
    pub fn run_qsa_page_place(
        jobs: *const i64,
        d: i32,
        n_jobs: i32,
        max_rows: i32,
        tile_r: i32,
        stream: *mut c_void,
    );
}
