//! FFI binding for QSA index page placement.
//!
//! Rotates a page's prepared block keys into the frame of the position it is
//! being placed at, and writes the scorer's channel-blocked staging in the same
//! pass. See `simple/qsa_page_place.cu`.
//!
//! ```text
//!   src   device f32[rows, d]        the page's keys, in the frame it was
//!                                    roped in
//!   dst   device f32[d/4, rows, 4]   this placement's staging
//!   delta = placement_base − roped_base, in tokens
//! ```
//!
//! Batched over a descriptor table: one job per (page, layer), because a
//! projection rebuilds every page of every attention layer at once.

use std::ffi::c_void;

/// i64 words per job: `{src, dst, rows, delta}`.
pub const PLACE_JOB_WORDS: usize = 4;

extern "C" {
    /// `tile_r` selects the row tile; `0` takes the tuned default. The explicit
    /// values exist for the sweep in `tests/qsa_page_place_bench.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_page_place(
        jobs: *const i64,
        cos_tab: *const f32,
        sin_tab: *const f32,
        d: i32,
        rope_dim: i32,
        n_jobs: i32,
        max_rows: i32,
        tile_r: i32,
        stream: *mut c_void,
    );
}
